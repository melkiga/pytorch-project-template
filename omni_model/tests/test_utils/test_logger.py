import pytest
from omni_model.src.utils.logger import Logger
from omni_model.src.data.dataset_helpers import _DATASET_TO_GROUP


@pytest.fixture
def loggerette():
    yield Logger


def test_loggerette(loggerette):
    log = loggerette(write=True)
    log("Hello")
    with pytest.raises(Exception):
        log("ERROR: I does not compute!", log_level=Logger.ERROR)
    log.log_value("train_epoch.epoch", 5, hide=False)
    log.log_dict(
        "_DATASET_TO_GROUP",
        _DATASET_TO_GROUP,
        hide=False,
        description="Mapping between dataset key to transform group.",
    )
