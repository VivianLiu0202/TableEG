from .error_generation import ErrorGenerationTask
from .error_detection import ErrorDetectionTask
from .error_correction import ErrorCorrectionTask

class TableTaskFactory:
    @staticmethod
    def get_table_task(task_type, sample_size=5):  # 新增 sample_size
        if task_type == "Error_Generation":
            return ErrorGenerationTask(sample_size=sample_size)  # 传入 sample_size
        elif task_type == "Error_Detection":
            return ErrorDetectionTask()
        elif task_type == "Error_Correction":
            return ErrorCorrectionTask()
        else:
            raise ValueError(f"Unknown task type: {task_type}")