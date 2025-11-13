
from roboflow import Roboflow
rf = Roboflow(api_key="HG9M6YJZpcCUgAQaKO9v")
project = rf.workspace("arakon").project("detection-base-hqaeg")
version = project.version(6)
dataset = version.download("yolov8")

print(f'Roboflow dataset downloaded to: {dataset.location}')
