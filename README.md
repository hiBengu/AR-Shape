# AR-Shape

Shape manipulation from camera input written in Python.

## How to Run the Project

### Prerequisites

- **Python 3.x** must be installed
- `pip` must be available (comes with Python 3)

---

### Install Dependencies

Only requirements.txt needed to install all required packages:

```bash
# Create a virtual environment
python3 -m venv venv

# Activate it
# On Linux/macOS:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

# Then install dependencies
pip install -r requirements.txt
pip install -r requirements.txt
```

Run below command
```
python AugmentedReality.py
```
![AR](https://github.com/hiBengu/AR-Shape/blob/main/ar.gif)


Object persistence option is buggy since other moving objects could effect the camera movement detection, but still can be tried out with below option
```
python AugmentedReality.py --persistence
```
![ARpersistence](https://github.com/hiBengu/AR-Shape/blob/main/ar_persistence.gif)

