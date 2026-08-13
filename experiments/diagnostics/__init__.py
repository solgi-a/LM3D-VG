"""
Checks that the implementation does what the paper describes.

    verify_eq7_distance_bias.py   measures the sign and magnitude of the distance bias
                                  in the fusion attention, against Eq. 7 as printed
    validate_scene_cache.py       end-to-end vs cached Acc@0.25, agreeing to 1e-4
    audit_scene_cache.py          model-free integrity + detector recall ceiling
    cached_eval_cpu.py            cached-path evaluation that never calls get_loss,
                                  so it runs on a machine with no CUDA

These verify correctness rather than producing a table, and apart from
``validate_scene_cache.py`` they all run on CPU in seconds.
"""
