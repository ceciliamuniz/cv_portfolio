# Module 7 — Part 1: Stereo Size Estimation

This part implements object-size estimation using calibrated stereo. It exposes a small Flask blueprint at `submissions/Module7/part1_stereosize/app.py` with endpoints:

- `GET /module7/` — UI page
- `POST /module7/api/stereo/estimate` — accepts left/right images and a binary mask, returns estimated object dimensions.

How to test locally:
1. Start the main app from the project root.
2. Open `http://localhost:5000/module7` and upload left/right images and a mask.
