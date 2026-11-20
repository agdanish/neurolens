"""Export the fine-tuned ViT backbone so NeuroLens can use the ViT model.

WHY THIS IS NEEDED
FundusViT-B16.weights.h5 only contains the 1.3M-parameter classification head.
`model.save_weights()` / ModelCheckpoint did not capture the ~86M-parameter
HuggingFace TFViTModel wrapped inside VisionTransformerFeatureExtractor, so the
backbone fine-tuned in Phase 2 of training was never saved. Until it is
exported, the dashboard excludes ViT from scoring instead of using the generic
ImageNet backbone (which does not match the head).

HOW TO USE (run in the SAME Colab/Jupyter session that trained the model, i.e.
while `feature_extractor` from `create_vit_model()` still holds the fine-tuned
weights -- after training, or after loading the best checkpoint into it):

    exec(open("export_vit_backbone.py").read())   # or paste the lines below
    export_vit_backbone(feature_extractor)

Then download the `vit_backbone/` folder (config.json + tf_model.h5) into
4-FundusViT-B16-VisionTransformer/ in this repo and commit it (*.h5 is tracked
by Git LFS). The app detects it automatically; no code change is needed.

If the training session is gone, the fine-tuned backbone cannot be recovered
from the head-only file -- ViT has to be retrained, this time also calling
export_vit_backbone() (e.g. from a callback on the best epoch).
"""

def export_vit_backbone(feature_extractor, out_dir="vit_backbone"):
    if getattr(feature_extractor, "vit", None) is None:
        raise ValueError("feature_extractor has no .vit backbone to export")
    feature_extractor.vit.save_pretrained(out_dir)  # writes config.json + tf_model.h5
    print(f"Saved fine-tuned ViT backbone to {out_dir}/ -- copy it into "
          "4-FundusViT-B16-VisionTransformer/ in the NeuroLens repo.")
