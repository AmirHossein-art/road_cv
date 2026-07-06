from ultralytics.models.sam import SAM3SemanticPredictor

overrides = dict(
    conf=0.25,
    task="segment",
    mode="predict",
    model="../sam3.pt",
    device="cuda:0",
    quantize=16,
    save=True,
)

predictor = SAM3SemanticPredictor(overrides=overrides)

image_path = "0004.760.jpg"

predictor.set_image(image_path)

results = predictor(
    text=[
        "concrete barrier",
        "road barrier",
        "jersey barrier",
        "median barrier",
        "guardrail",
    ]
)

print("Done.")