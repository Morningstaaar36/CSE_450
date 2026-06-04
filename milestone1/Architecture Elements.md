# Architecture Elements

## **Loader layer**
*   parse point cloud / 3DGS
*   create analysis cloud
*   compute cloud stats
## **Alignment engine**
Submodules:
*   OutlierCleaner
*   ContentTypeClassifier
*   GeometryAxisEstimator
*   MLAxisEstimator
*   AlignmentFusion
*   ConfidenceScorer
Output:
*   4x4 alignment transform
*   confidence
*   mode used
*   ambiguity flags
## **Renderer integration**
*   apply alignment to scene root transform, or
*   bake into splat data once after load
Prefer a **scene root transform first**, then optionally bake later.
It is safer while iterating.
## **UX layer**
*   Auto Align button
*   Align Up Only
*   Rotate 90°
*   Flip Front/Back
*   Save Alignment