# CLUSRec
CLothes and hUman Silhoutte Recognition

# Project Structure


```
|-- CLUSRec/
|   |-- cnn/
|   |   |-- __init__.py
|   |   |-- smallervggnet.py
|   |-- clothes_recognition
|   |   |-- classify.py
|   |   |-- clothes.model
|   |   |-- mlb.pickle
|   |   |-- plot.png
|   |   |-- train.py
|   |-- human_recognition
|   |   |-- classify.py
|   |   |-- human.model
|   |   |-- mlb.pickle
|   |   |-- plot.png
|   |   |-- train.py
|   |-- datasets/ [total_images]
|   |   |-- clothes/ [total_clothes]
|   |   |   |-- black_jeans/ [total_black_jeans]
|   |   |   |-- blue_jeans/ [total_blue_jeans]
|   |   |   |-- ...
|   |   |-- human/ [total_human]
|   |   |   |-- single_person/ [total_single_person]
|   |   |   |-- multiple_people/ [total_multiple_people]
|   |   |   |-- other/ [total_other]
|   |-- tests/
|   |   |-- test_01.jpg
|   |   |-- test_n.jpg
```
