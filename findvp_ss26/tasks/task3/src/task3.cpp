#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

#include <pcl/PolygonMesh.h>
#include <pcl/conversions.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

using namespace std;
namespace fs = std::filesystem;

#ifndef TASKS_ROOT_DIR
#define TASKS_ROOT_DIR "."
#endif

static const fs::path kTasksRoot = fs::path(TASKS_ROOT_DIR);

static constexpr double PI = 3.14159265358979323846;
static double deg2rad(double d) { return d * PI / 180.0; }
static double wrap360(double a) { while (a < 0) a += 360; while (a >= 360) a -= 360; return a; }
static double clampd(double v, double lo, double hi) { return max(lo, min(v, hi)); }

static cv::Mat normalize_target_to_render_frame(const cv::Mat& src, int width, int height) {
    if (src.empty()) return src;
    if (src.cols == width && src.rows == height) return src.clone();

    const double scale = min(
        static_cast<double>(width) / static_cast<double>(src.cols),
        static_cast<double>(height) / static_cast<double>(src.rows));

    const int resized_width = max(1, static_cast<int>(std::round(src.cols * scale)));
    const int resized_height = max(1, static_cast<int>(std::round(src.rows * scale)));

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(resized_width, resized_height), 0.0, 0.0, cv::INTER_AREA);

    cv::Mat canvas(height, width, src.type(), cv::Scalar(255, 255, 255));
    const int x = (width - resized_width) / 2;
    const int y = (height - resized_height) / 2;
    resized.copyTo(canvas(cv::Rect(x, y, resized_width, resized_height)));
    return canvas;
}

// Step 3: Robot is not used. It is kept only to follow the supervisor draft.
class Robot {};

// One 6-DoF camera viewpoint: T_world_camera + readable parameters.
class View {
public:
    Eigen::Matrix4d pose_6d = Eigen::Matrix4d::Identity();
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    double radius = 0.0, azimuth_deg = 0.0, elevation_deg = 0.0, roll_deg = 0.0;
    double score = numeric_limits<double>::infinity();

    // Backward-compatible wrapper for the professor draft typo.
    void compute_pose_from_positon_and_object_center(Eigen::Vector3d p, Eigen::Vector3d c) {
        compute_pose_from_position_and_object_center(p, c, 0.0);
    }

    // Level 4: look-at 6-DoF pose. Camera position is sampled; orientation looks at object center.
    void compute_pose_from_position_and_object_center(
        const Eigen::Vector3d& camera_pos,
        const Eigen::Vector3d& object_center,
        double roll = 0.0)
    {
        position = camera_pos;
        roll_deg = roll;

        Eigen::Vector3d Z = object_center - camera_pos;  // optical axis toward object
        if (Z.norm() < 1e-9) Z = Eigen::Vector3d(0, 0, -1);
        Z.normalize();

        Eigen::Vector3d up(0, 0, 1);
        if (abs(Z.dot(up)) > 0.98) up = Eigen::Vector3d(0, 1, 0);

        Eigen::Vector3d X = (-Z).cross(up).normalized();
        Eigen::Vector3d Y = X.cross(-Z).normalized();

        const double r = deg2rad(roll);
        Eigen::Vector3d Xr = cos(r) * X + sin(r) * Y;
        Eigen::Vector3d Yr = -sin(r) * X + cos(r) * Y;

        pose_6d = Eigen::Matrix4d::Identity();
        pose_6d.block<3, 1>(0, 0) = Xr;
        pose_6d.block<3, 1>(0, 1) = Yr;
        pose_6d.block<3, 1>(0, 2) = Z;
        pose_6d.block<3, 1>(0, 3) = camera_pos;
    }
};

class Perception {
public:
    int width = 320, height = 240;        // faster for search than 640x480
    double fov_x = 0.95, fov_y = 0.75;
    Eigen::Matrix3f intrinsics;
    Eigen::Vector3d object_center = Eigen::Vector3d::Zero(); // after normalization
    Eigen::Vector3d original_object_center = Eigen::Vector3d::Zero();
    double original_object_radius = 1.0;

    pcl::PolygonMesh::Ptr mesh_ply;
    pcl::visualization::PCLVisualizer::Ptr viewer;

    explicit Perception(const string& object_path) {
        setup_camera();
        load_and_normalize_object(object_path);
        create_viewer();
    }

    ~Perception() {
        cout << "Perception destruct successfully!" << endl;
    }

    void setup_camera() {
        const double fx = width / (2.0 * tan(fov_x / 2.0));
        const double fy = height / (2.0 * tan(fov_y / 2.0));
        intrinsics << fx, 0, width / 2.0,
                      0, fy, height / 2.0,
                      0, 0, 1;
        cout << "Camera: " << width << "x" << height << endl;
    }

    void load_and_normalize_object(const string& object_path) {
        mesh_ply.reset(new pcl::PolygonMesh);
        if (pcl::io::loadPLYFile(object_path, *mesh_ply) < 0 ||
            mesh_ply->cloud.data.empty() || mesh_ply->polygons.empty()) {
            cerr << "Failed to load object: " << object_path << endl;
            exit(1);
        }

        pcl::PointCloud<pcl::PointXYZ>::Ptr vertices(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromPCLPointCloud2(mesh_ply->cloud, *vertices);

        Eigen::Vector3d c(0, 0, 0);
        for (const auto& p : vertices->points) c += Eigen::Vector3d(p.x, p.y, p.z);
        c /= static_cast<double>(vertices->points.size());
        original_object_center = c;

        double radius = 0.0;
        for (const auto& p : vertices->points)
            radius = max(radius, (Eigen::Vector3d(p.x, p.y, p.z) - c).norm());
        if (radius < 1e-9) {
            cerr << "Invalid object radius." << endl;
            exit(1);
        }
        original_object_radius = radius;

        const double scale = 1.0 / radius;
        const int step = mesh_ply->cloud.point_step;
        const int ox = mesh_ply->cloud.fields[0].offset;
        const int oy = mesh_ply->cloud.fields[1].offset;
        const int oz = mesh_ply->cloud.fields[2].offset;

        for (size_t i = 0; i < mesh_ply->cloud.data.size(); i += step) {
            float x, y, z;
            memcpy(&x, &mesh_ply->cloud.data[i + ox], sizeof(float));
            memcpy(&y, &mesh_ply->cloud.data[i + oy], sizeof(float));
            memcpy(&z, &mesh_ply->cloud.data[i + oz], sizeof(float));

            x = static_cast<float>((x - c.x()) * scale);
            y = static_cast<float>((y - c.y()) * scale);
            z = static_cast<float>((z - c.z()) * scale);

            memcpy(&mesh_ply->cloud.data[i + ox], &x, sizeof(float));
            memcpy(&mesh_ply->cloud.data[i + oy], &y, sizeof(float));
            memcpy(&mesh_ply->cloud.data[i + oz], &z, sizeof(float));
        }

        object_center = Eigen::Vector3d(0, 0, 0);
        cout << "Object normalized. Assumed centroid: " << object_center.transpose() << endl;
    }

    Eigen::Vector3d normalized_to_original(const Eigen::Vector3d& p) const {
        return p * original_object_radius + original_object_center;
    }

    void create_viewer() {
        viewer.reset(new pcl::visualization::PCLVisualizer("Render Viewer"));
        viewer->setBackgroundColor(255, 255, 255);
        viewer->initCameraParameters();
        viewer->addPolygonMesh(*mesh_ply, "object");
        viewer->setSize(width, height);
        viewer->spinOnce(10);
    }

    cv::Mat render(const View& view, const string& save_path = "") {
        viewer->setCameraParameters(intrinsics, view.pose_6d.cast<float>());
        viewer->spinOnce(10);

        string tmp = "tmp_render.png";
        viewer->saveScreenshot(tmp);
        cv::Mat img = cv::imread(tmp, cv::IMREAD_COLOR);
        remove(tmp.c_str());

        if (img.empty()) return img;
        if (img.cols >= width && img.rows >= height && (img.cols != width || img.rows != height))
            img = img(cv::Rect(img.cols - width, img.rows - height, width, height)).clone();
        if (img.cols != width || img.rows != height)
            cv::resize(img, img, cv::Size(width, height));

        if (!save_path.empty()) {
            fs::create_directories(fs::path(save_path).parent_path());
            cv::imwrite(save_path, img);
        }
        return img;
    }
};

// Level 4: object-centered hemisphere sampling.
static View make_view(const Eigen::Vector3d& c, double radius, double az, double el, double roll) {
    const double th = deg2rad(az), ph = deg2rad(el);
    Eigen::Vector3d p;
    p.x() = c.x() + radius * cos(th) * cos(ph);
    p.y() = c.y() + radius * sin(th) * cos(ph);
    p.z() = c.z() + radius * sin(ph);

    View v;
    v.radius = radius;
    v.azimuth_deg = wrap360(az);
    v.elevation_deg = el;
    v.roll_deg = roll;
    v.compute_pose_from_position_and_object_center(p, c, roll);
    return v;
}

static vector<View> generate_views(
    const Eigen::Vector3d& c,
    const vector<double>& radii,
    const vector<double>& azimuths,
    const vector<double>& elevations,
    const vector<double>& rolls)
{
    vector<View> views;
    for (double r : radii)
        for (double az : azimuths)
            for (double el : elevations)
                for (double roll : rolls)
                    views.push_back(make_view(c, r, az, el, roll));
    return views;
}

static vector<View> refine_views(const Eigen::Vector3d& c, const View& best,
                                 double dr, double daz, double del, double droll) {
    return generate_views(
        c,
        {max(0.5, best.radius - dr), best.radius, best.radius + dr},
        {wrap360(best.azimuth_deg - daz), best.azimuth_deg, wrap360(best.azimuth_deg + daz)},
        {clampd(best.elevation_deg - del, 5, 85), best.elevation_deg, clampd(best.elevation_deg + del, 5, 85)},
        {clampd(best.roll_deg - droll, -45, 45), best.roll_deg, clampd(best.roll_deg + droll, -45, 45)});
}

struct SearchResult {
    View best;
    int searched = 0;
    double time_sec = 0.0;
};

struct EvaluatedViewRecord {
    int order = 0;
    int stage_id = 0;
    string stage_name;
    View view;
};

static int stage_id_from_name(const string& stage_name) {
    if (stage_name.find("Stage 0") != string::npos) return 0;
    if (stage_name.find("Stage 1") != string::npos) return 1;
    if (stage_name.find("Stage 2") != string::npos) return 2;
    return 3;
}

static array<unsigned char, 3> color_for_stage(int stage_id) {
    switch (stage_id) {
        case 0: return {255, 80, 80};
        case 1: return {255, 180, 0};
        case 2: return {80, 200, 120};
        default: return {80, 140, 255};
    }
}

class View_Planning_Simulator {
public:
    Perception* perception_simulator;
    cv::Mat target_image, target_small, target_gray, target_gray_full, target_descriptors;
    vector<cv::KeyPoint> target_keypoints;
    cv::Ptr<cv::SIFT> sift;
    vector<View> view_space;
    vector<View> selected_views;
    vector<EvaluatedViewRecord> evaluated_views;

    View_Planning_Simulator(Perception* p, const cv::Mat& target, const vector<View>& views = {})
        : perception_simulator(p), view_space(views)
    {
        if (target.empty()) {
            cerr << "Target image is empty." << endl;
            exit(1);
        }
        target_image = normalize_target_to_render_frame(target, p->width, p->height);
        cv::resize(target_image, target_small, cv::Size(160, 120));
        cv::cvtColor(target_small, target_gray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(target_image, target_gray_full, cv::COLOR_BGR2GRAY);
        sift = cv::SIFT::create();
        sift->detectAndCompute(target_gray_full, cv::noArray(), target_keypoints, target_descriptors);
    }

    cv::Mat render_view_image(const View& view) { return perception_simulator->render(view); }

    double sift_similarity(const cv::Mat& gray_full) {
        if (target_descriptors.empty()) return -1.0;

        vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        sift->detectAndCompute(gray_full, cv::noArray(), keypoints, descriptors);
        if (descriptors.empty() || keypoints.size() < 4 || target_keypoints.size() < 4) {
            return -1.0;
        }

        cv::BFMatcher matcher(cv::NORM_L2);
        vector<vector<cv::DMatch>> knn_matches;
        matcher.knnMatch(target_descriptors, descriptors, knn_matches, 2);

        vector<cv::DMatch> good_matches;
        good_matches.reserve(knn_matches.size());
        for (const auto& pair : knn_matches) {
            if (pair.size() < 2) continue;
            if (pair[0].distance < 0.75f * pair[1].distance) {
                good_matches.push_back(pair[0]);
            }
        }

        if (good_matches.size() < 4) return 0.0;

        vector<cv::Point2f> src_pts;
        vector<cv::Point2f> dst_pts;
        src_pts.reserve(good_matches.size());
        dst_pts.reserve(good_matches.size());
        for (const auto& match : good_matches) {
            src_pts.push_back(target_keypoints[match.queryIdx].pt);
            dst_pts.push_back(keypoints[match.trainIdx].pt);
        }

        cv::Mat inlier_mask;
        cv::findHomography(src_pts, dst_pts, cv::RANSAC, 3.0, inlier_mask);
        if (inlier_mask.empty()) return 0.0;

        int inliers = 0;
        for (int i = 0; i < inlier_mask.rows; ++i) {
            inliers += inlier_mask.at<unsigned char>(i, 0) != 0;
        }

        const double denom = static_cast<double>(max(1, min(target_keypoints.size(), keypoints.size())));
        return clampd(static_cast<double>(inliers) / denom, 0.0, 1.0);
    }

    // Level 5: lower score means closer to target.
    double score_image(const cv::Mat& img) {
        if (img.empty()) return numeric_limits<double>::infinity();

        cv::Mat small, gray, gray_full;
        cv::resize(img, small, target_small.size());
        cv::cvtColor(small, gray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(img, gray_full, cv::COLOR_BGR2GRAY);

        cv::Mat corr;
        cv::matchTemplate(target_gray, gray, corr, cv::TM_CCOEFF_NORMED);
        double zncc = -1.0;
        cv::minMaxLoc(corr, nullptr, &zncc, nullptr, nullptr);

        cv::Mat diff;
        cv::absdiff(target_small, small, diff);
        cv::Scalar m = cv::mean(diff);
        double color_mae = (m[0] + m[1] + m[2]) / (3.0 * 255.0);

        const double sift_score = sift_similarity(gray_full);
        if (sift_score < 0.0) {
            return 0.7 * (1.0 - zncc) + 0.3 * color_mae;
        }

        return 0.55 * (1.0 - zncc) + 0.35 * (1.0 - sift_score) + 0.10 * color_mae;
    }

    bool is_target(const View& view) {
        return score_image(render_view_image(view)) < 0.015;
    }

    View search_next_view() {
        SearchResult r = evaluate(view_space, "single search");
        return r.best;
    }

    SearchResult evaluate(const vector<View>& candidates, const string& stage) {
        cout << "\n" << stage << " | candidates: " << candidates.size() << endl;
        SearchResult result;
        const int stage_id = stage_id_from_name(stage);
        auto t0 = chrono::steady_clock::now();

        for (View v : candidates) {
            v.score = score_image(render_view_image(v));
            evaluated_views.push_back(EvaluatedViewRecord{
                static_cast<int>(evaluated_views.size()),
                stage_id,
                stage,
                v
            });
            result.searched++;
            if (v.score < result.best.score) {
                result.best = v;
                cout << "best score=" << v.score
                     << " r=" << v.radius
                     << " az=" << v.azimuth_deg
                     << " el=" << v.elevation_deg
                     << " roll=" << v.roll_deg << endl;
            }
        }

        auto t1 = chrono::steady_clock::now();
        result.time_sec = chrono::duration<double>(t1 - t0).count();
        selected_views.push_back(result.best);
        return result;
    }

    SearchResult loop(const Eigen::Vector3d& object_center) {
        SearchResult total;

        // Stage 0: coarse global hemisphere search.
        vector<View> stage0 = generate_views(
            object_center,
            {2.2, 2.8, 3.4},
            {0, 45, 90, 135, 180, 225, 270, 315},
            {20, 40, 60, 75},
            {-15, 0, 15});
        SearchResult r0 = evaluate(stage0, "Stage 0: coarse hemisphere search");

        // Stage 1: medium refinement around best.
        vector<View> stage1 = refine_views(object_center, r0.best, 0.30, 15.0, 10.0, 8.0);
        SearchResult r1 = evaluate(stage1, "Stage 1: local refinement");

        // Stage 2: fine refinement around best.
        vector<View> stage2 = refine_views(object_center, r1.best, 0.12, 5.0, 4.0, 3.0);
        SearchResult r2 = evaluate(stage2, "Stage 2: fine refinement");

        total.best = r2.best;
        total.searched = r0.searched + r1.searched + r2.searched;
        total.time_sec = r0.time_sec + r1.time_sec + r2.time_sec;
        return total;
    }
};

static void save_evaluated_views_csv(
    const fs::path& path,
    const vector<EvaluatedViewRecord>& evaluated_views,
    const Perception& perception)
{
    ofstream out(path.string());
    out << "order,stage_id,stage_name,score,"
        << "x_normalized,y_normalized,z_normalized,"
        << "x_original,y_original,z_original,"
        << "radius,azimuth_deg,elevation_deg,roll_deg\n";

    for (const auto& record : evaluated_views) {
        const Eigen::Vector3d p_original = perception.normalized_to_original(record.view.position);
        out << record.order << ","
            << record.stage_id << ","
            << '"' << record.stage_name << '"' << ","
            << record.view.score << ","
            << record.view.position.x() << ","
            << record.view.position.y() << ","
            << record.view.position.z() << ","
            << p_original.x() << ","
            << p_original.y() << ","
            << p_original.z() << ","
            << record.view.radius << ","
            << record.view.azimuth_deg << ","
            << record.view.elevation_deg << ","
            << record.view.roll_deg << "\n";
    }
}

static void save_evaluated_views_ply(
    const fs::path& path,
    const vector<EvaluatedViewRecord>& evaluated_views,
    const Perception& perception)
{
    ofstream out(path.string());
    out << "ply\n";
    out << "format ascii 1.0\n";
    out << "comment Task 3 searched viewpoints in original mesh coordinates\n";
    out << "element vertex " << evaluated_views.size() << "\n";
    out << "property float x\n";
    out << "property float y\n";
    out << "property float z\n";
    out << "property uchar red\n";
    out << "property uchar green\n";
    out << "property uchar blue\n";
    out << "end_header\n";

    for (const auto& record : evaluated_views) {
        const Eigen::Vector3d p_original = perception.normalized_to_original(record.view.position);
        const auto color = color_for_stage(record.stage_id);
        out << p_original.x() << " "
            << p_original.y() << " "
            << p_original.z() << " "
            << static_cast<int>(color[0]) << " "
            << static_cast<int>(color[1]) << " "
            << static_cast<int>(color[2]) << "\n";
    }
}

static void save_evaluated_views_trajectory_obj(
    const fs::path& path,
    const vector<EvaluatedViewRecord>& evaluated_views,
    const Perception& perception)
{
    ofstream out(path.string());
    out << "# Task 3 searched viewpoint trajectory in original mesh coordinates\n";
    for (const auto& record : evaluated_views) {
        const Eigen::Vector3d p_original = perception.normalized_to_original(record.view.position);
        out << "v " << p_original.x() << " "
            << p_original.y() << " "
            << p_original.z() << "\n";
    }
    if (evaluated_views.size() >= 2) {
        out << "l";
        for (size_t i = 0; i < evaluated_views.size(); ++i) {
            out << " " << (i + 1);
        }
        out << "\n";
    }
}

static void save_result(
    const string& object,
    const SearchResult& r,
    const View_Planning_Simulator& planner,
    Perception* p,
    int test_id,
    bool track_points)
{
    const fs::path dir = kTasksRoot / "task3" / "results" / object;
    const fs::path img_dir = kTasksRoot / "task3" / "result_images" / object;
    fs::create_directories(dir);
    fs::create_directories(img_dir);
    const Eigen::Vector3d best_position_original = p->normalized_to_original(r.best.position);

    ofstream out((dir / ("test_" + to_string(test_id) + ".txt")).string());
    out << "searched_viewpoints " << r.searched << "\n";
    out << "computation_time_sec " << r.time_sec << "\n";
    out << "best_score " << r.best.score << "\n";
    out << "radius " << r.best.radius << "\n";
    out << "azimuth_deg " << r.best.azimuth_deg << "\n";
    out << "elevation_deg " << r.best.elevation_deg << "\n";
    out << "roll_deg " << r.best.roll_deg << "\n";
    out << "position " << r.best.position.transpose() << "\n";
    out << "position_original " << best_position_original.transpose() << "\n";
    out << "pose_6d\n" << r.best.pose_6d << "\n";

    p->render(r.best, (img_dir / ("best_view_test_" + to_string(test_id) + ".png")).string());
    if (track_points) {
        save_evaluated_views_csv(
            dir / ("searched_views_test_" + to_string(test_id) + ".csv"),
            planner.evaluated_views,
            *p);
        save_evaluated_views_ply(
            dir / ("searched_views_test_" + to_string(test_id) + ".ply"),
            planner.evaluated_views,
            *p);
        save_evaluated_views_trajectory_obj(
            dir / ("searched_trajectory_test_" + to_string(test_id) + ".obj"),
            planner.evaluated_views,
            *p);
    }
}

static void task3_with_target(const string& object, const string& target_path, bool track_points) {
    Perception* p = new Perception((kTasksRoot / "3d_models" / (object + ".ply")).string());
    cout << "Using user target image: " << fs::absolute(target_path) << endl;
    cv::Mat target = cv::imread(target_path, cv::IMREAD_COLOR);
    if (target.empty()) {
        cerr << "Could not read target image: " << target_path << endl;
        delete p;
        exit(1);
    }
    cout << "Original target size: " << target.cols << "x" << target.rows << endl;
    if (target.cols != p->width || target.rows != p->height) {
        cout << "Normalizing target to render frame: "
             << p->width << "x" << p->height
             << " with aspect-ratio-preserving padding" << endl;
    }

    View_Planning_Simulator planner(p, target);
    SearchResult r = planner.loop(p->object_center);

    cout << "\nFinal result: score=" << r.best.score
         << " searched=" << r.searched
         << " time=" << r.time_sec << " s" << endl;
    save_result(object, r, planner, p, 0, track_points);
    delete p;
}

static void task3_self_test(const string& object, bool track_points) {
    Perception* p = new Perception((kTasksRoot / "3d_models" / (object + ".ply")).string());

    // Hidden target for debugging only.
    View target_view = make_view(p->object_center, 2.55, 52.0, 26.0, 2.0);
    cv::Mat target = p->render(target_view);

    View_Planning_Simulator planner(p, target);
    SearchResult r = planner.loop(p->object_center);

    cout << "\nHidden target: r=2.55 az=52 el=26 roll=2" << endl;
    cout << "Final result: score=" << r.best.score
         << " searched=" << r.searched
         << " time=" << r.time_sec << " s" << endl;
    save_result(object, r, planner, p, 0, track_points);
    delete p;
}

int main(int argc, char** argv) {
    string object = "obj_000020";
    string target_path;
    bool track_points = false;
    vector<string> positional_args;

    for (int i = 1; i < argc; ++i) {
        const string arg = argv[i];
        if (arg == "track_points") {
            track_points = true;
            continue;
        }
        positional_args.push_back(arg);
    }

    if (!positional_args.empty()) object = positional_args[0];
    if (positional_args.size() >= 2) target_path = positional_args[1];

    cout << "Step 3: 6-DoF Viewpoint and Planning" << endl;
    cout << "Level 4: object-centered SE(3) viewpoint generation" << endl;
    cout << "Level 5: coarse-to-fine next-best-view search" << endl;
    if (track_points) {
        cout << "Tracking searched viewpoints is enabled." << endl;
    }

    if (!target_path.empty()) {
        task3_with_target(object, target_path, track_points);
    } else {
        cout << "No target image provided. Running self-test." << endl;
        cout << "Usage: ./task3 obj_000020 /path/to/target.png [track_points]" << endl;
        task3_self_test(object, track_points);
    }
    return 0;
}
