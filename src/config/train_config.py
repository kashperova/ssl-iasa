from typing import Dict, Any


class BaseTrainConfig:
    epochs: int = None
    train_batch_size: int = None
    eval_batch_size: int = None
    train_test_split: float = None

    @property
    def params(self) -> Dict[str, Any]:
        attrs = set(self.__class__.__dict__.keys()) - set(self.__dict__.keys())
        return {attr: getattr(self, attr) for attr in attrs}


class SLConfig(BaseTrainConfig):
    teacher: BaseTrainConfig = None
    student: BaseTrainConfig = None

    def set(self, teacher: Dict[str, Any], student: Dict[str, Any]) -> None:
        for k, v in teacher.items():
            setattr(self.teacher, k, v)

        for k, v in student.items():
            setattr(self.student, k, v)
