from roboflow import Roboflow

API_KEY = "mh0BpqEfORKLMgzhkW1l"
ROBOFLOW_PROJECT = "open-cv-athd6"


def download_dataset():
    rf = Roboflow(api_key=API_KEY)
    project = rf.workspace().project(ROBOFLOW_PROJECT)
    dataset = project.version(1).download("yolov5pytorch")
    print("Dataset: ", dataset)


if __name__ == "__main__":
    download_dataset()
