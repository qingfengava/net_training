"""Keep Ultralytics from phoning home or downloading weights during training and export.

scripts/env.sh exports YOLO_OFFLINE and YOLO_AUTOINSTALL. apply() adds the two things env vars cannot do:
- turns off the `sync` setting (anonymous usage events);
- replaces trainer.check_amp. The real check does YOLO("yolo26n.pt"), i.e. it downloads a pickle from GitHub
  and loads it. Skipping it means AMP is trusted blindly, so M0 verifies AMP training on this GPU instead
  (nt/smoke_m0.py) and cloud runs should repeat that check once per new GPU type.
"""


def apply() -> None:
    import ultralytics.engine.trainer as trainer
    from ultralytics import settings

    settings.update({"sync": False})
    trainer.check_amp = lambda model: True
