import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from ultralytics import YOLO
import argparse


# Словарь аугментаций для разных уровней
AUGMENTATION_PRESETS = {
    "none": {},
    "custom": {
            'fliplr':0.5,    # Для чистоты эксперимента берем такие же, как и для FCOS
             }
}

def train_model(data_path, epochs=100, imgsz=640, batch=8, model_name='yolov8n.pt',
                augment_strength="none", experiment_name='rsp_baseline'):

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Файл {data_path} не найден!")


    model = YOLO(model_name)

    augmentations = AUGMENTATION_PRESETS.get(augment_strength.lower(), {})


    model.train(
        data=data_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        lr0=0.001,
        lrf=0.01,
        cos_lr=True,
        # augment_strength=augment_strength,
        name=experiment_name,
        patience=10,
        seed=42 ,
        **augmentations,
        amp=False # Временное решение, пока не обновлю окружение
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Обучение модели YOLOv8 для детекции жестов "Камень-Ножницы-Бумага"')

    # Добавьте аргументы командной строки
    parser.add_argument('--data', type=str, default='data.yaml', help='Путь к data.yaml')
    parser.add_argument('--epochs', type=int, default=100, help='Количество эпох')
    parser.add_argument('--imgsz', type=int, default=640, help='Размер изображения')
    parser.add_argument('--batch', type=int, default=8, help='Размер батча')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='Модель нейронной сети')
    parser.add_argument("--augment", type=str, default="none",
                        choices=["none", "custom"],
                        help="Уровень аугментации данных")
    parser.add_argument('--name', type=str, default='rsp_baseline', help='Название эксперимента')


    args = parser.parse_args()


    train_model(
        data_path=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        model_name=args.model,
        augment_strength=args.augment,
        experiment_name=args.name
    )