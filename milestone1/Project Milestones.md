# Project Milestones

## **Milestone 1 — Working baseline app**
Goal: app loads 3DGS and auto-aligns many easy cases
Deliverables:
*   preprocessing/downsampling
*   scene vs object classification
*   geometry-based alignment
*   preview/apply/reset/manual tweak
*   axis gizmo in renderer
Success criteria:
*   indoor scenes align upright reliably
*   obvious objects get roughly aligned
*   no meaningful FPS drop after load

* * *

## **Milestone 2 — Research-to-product bridge**
Goal: adapt current EQR work into a useful ML assist module
Deliverables:
*   dataset split for object vs scene
*   real 3DGS-derived cloud set
*   model for up-axis and yaw ambiguity
*   confidence output
*   benchmark against geometry baseline
Success criteria:
*   ML beats baseline on ambiguous objects
*   stable predictions on noisy point clouds
*   confidence correlates with correctness

* * *

## **Milestone 3 — Hybrid fusion engine**
Goal: combine geometry and ML robustly
Deliverables:
*   fusion rules
*   fallback logic
*   ambiguity detection
*   user correction logging
Success criteria:
*   lower manual correction rate than geometry-only
*   fewer catastrophic misalignments
*   better behavior on mixed-quality captures

* * *

## **Milestone 4 — Production integration**
Goal: make alignment part of the normal rendering flow
Deliverables:
*   transform applied consistently to splats and camera reset
*   saved aligned state
*   export in aligned coordinates
*   Android/Vulkan optimization
Success criteria:
*   alignment works automatically on load
*   reset camera and auto camera respect aligned axes
*   selection/edit/export all operate in aligned frame