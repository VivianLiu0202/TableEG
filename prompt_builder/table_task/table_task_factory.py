from .error_generation import ErrorGenerationTask
from .error_detection import ErrorDetectionTask
from .error_correction import ErrorCorrectionTask

class TableTaskFactory:
    """Factory class for retrieving table-related task instances."""

    @staticmethod
    def get_table_task(task_type: str, sample_size: int = 5):
        """Retrieves the appropriate table task instance.

        Args:
            task_type (str): Type of task to retrieve. Must be one of:
                - "Error_Generation"
                - "Error_Detection"
                - "Error_Correction"
            sample_size (int, optional): Number of samples for error generation (default: 5).

        Returns:
            BaseTableTask: An instance of the corresponding table task class.

        Raises:
            ValueError: If the provided `task_type` is invalid.
        """
        task_mapping = {
            "Error_Generation": ErrorGenerationTask(sample_size=sample_size),
            "Error_Detection": ErrorDetectionTask(),
            "Error_Correction": ErrorCorrectionTask()
        }

        if task_type not in task_mapping:
            raise ValueError(f"Invalid task type: {task_type}. Expected one of {list(task_mapping.keys())}.")

        return task_mapping[task_type]