# Tasks Directory

This directory contains the task implementations organized by task:

- `task1/src/task1.cpp`
- `task3/src/task3.cpp`

Shared assets live here as well:

- `3d_models/`
- `view_space/`

Build from the repository root:

```bash
cmake -S tasks -B tasks/build
cmake --build tasks/build -j4
```

Generated binaries are written to:

```bash
tasks/build/bin/
```
