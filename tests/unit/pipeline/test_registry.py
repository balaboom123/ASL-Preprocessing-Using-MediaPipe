"""Tests for the component registry system."""

from signdata.registry import (
    DATASET_REGISTRY,
    PROCESSOR_REGISTRY,
    register_dataset,
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


class TestRealRegistrations:
    """Verify all real registrations exist after importing modules."""

    def test_datasets_registered(self):
        import signdata.datasets  # noqa: F401

        assert "youtube_asl" in DATASET_REGISTRY
        assert "how2sign" in DATASET_REGISTRY
        assert "openasl" in DATASET_REGISTRY
        assert "wlasl" in DATASET_REGISTRY
        assert "msasl" in DATASET_REGISTRY
        assert "csl" in DATASET_REGISTRY
        assert "slovo" in DATASET_REGISTRY
        assert len(DATASET_REGISTRY) >= 7

    def test_processors_registered(self):
        import signdata.processors  # noqa: F401

        expected = ["video2pose", "video2crop"]
        for name in expected:
            assert name in PROCESSOR_REGISTRY, f"Processor '{name}' not registered"
        assert len(PROCESSOR_REGISTRY) >= 2
