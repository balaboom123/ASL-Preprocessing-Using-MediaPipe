"""Tests for the component registry system."""

from signdata.registry import (
    DATASET_REGISTRY,
    OUTPUT_REGISTRY,
    POST_PROCESSOR_REGISTRY,
    PROCESSOR_REGISTRY,
    register_dataset,
    register_output,
    register_post_processor,
    register_processor,
)


class TestRegisterDataset:
    def test_adds_class_to_registry(self):
        @register_dataset("_test_ds")
        class _TestDS:
            pass

        assert "_test_ds" in DATASET_REGISTRY
        assert DATASET_REGISTRY["_test_ds"] is _TestDS

        # Clean up
        del DATASET_REGISTRY["_test_ds"]


class TestRegisterProcessor:
    def test_adds_class_to_registry(self):
        @register_processor("_test_proc")
        class _TestProc:
            pass

        assert "_test_proc" in PROCESSOR_REGISTRY
        assert PROCESSOR_REGISTRY["_test_proc"] is _TestProc

        del PROCESSOR_REGISTRY["_test_proc"]


class TestRegisterPostProcessor:
    def test_adds_class_to_registry(self):
        @register_post_processor("_test_pp")
        class _TestPP:
            pass

        assert "_test_pp" in POST_PROCESSOR_REGISTRY
        assert POST_PROCESSOR_REGISTRY["_test_pp"] is _TestPP

        del POST_PROCESSOR_REGISTRY["_test_pp"]


class TestRegisterOutput:
    def test_adds_class_to_registry(self):
        @register_output("_test_out")
        class _TestOut:
            pass

        assert "_test_out" in OUTPUT_REGISTRY
        assert OUTPUT_REGISTRY["_test_out"] is _TestOut

        del OUTPUT_REGISTRY["_test_out"]


class TestRealRegistrations:
    """Verify all real registrations exist after importing modules."""

    def test_datasets_registered(self):
        import signdata.datasets  # noqa: F401

        assert "youtube_asl" in DATASET_REGISTRY
        assert "how2sign" in DATASET_REGISTRY
        assert "openasl" in DATASET_REGISTRY
        assert "wlasl" in DATASET_REGISTRY
        assert len(DATASET_REGISTRY) >= 4

    def test_processors_registered(self):
        import signdata.processors  # noqa: F401

        expected = ["video2pose", "video2crop"]
        for name in expected:
            assert name in PROCESSOR_REGISTRY, f"Processor '{name}' not registered"
        assert len(PROCESSOR_REGISTRY) >= 2

    def test_post_processors_registered(self):
        import signdata.post_processors  # noqa: F401

        assert "normalize" in POST_PROCESSOR_REGISTRY
        assert len(POST_PROCESSOR_REGISTRY) >= 1

    def test_outputs_registered(self):
        import signdata.output  # noqa: F401

        assert "webdataset" in OUTPUT_REGISTRY
        assert len(OUTPUT_REGISTRY) >= 1
