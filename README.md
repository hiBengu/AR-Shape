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
python -m venv venv

# Activate it
# On Linux/macOS:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

# Then install dependencies
pip install -r requirements.txt
pip install -r requirements.txt
```

Keybindings for the project
```
ESC - close program
e E - Edit Mode
i I - Interact Mode
u U - Undo last drawn object
c C - Clear all objects 
```

Run below command to run the project
Camera option could change on device, run below command if camera feed is not captured.
```
python AugmentedReality.py
or
python AugmentedReality.py --cam 0
```
![AR](https://github.com/hiBengu/AR-Shape/blob/main/ar.gif)


Object persistence option is buggy since other moving objects could effect the camera movement detection, but still can be tried out with below option
```
python AugmentedReality.py --persistence
```
![ARpersistence](https://github.com/hiBengu/AR-Shape/blob/main/ar_persistence.gif)

I used AI for below steps during the assignment.
    - Deciding on the initial steps of the project, which libraries can be used, how many stages I should divide the project.
    - Details on the hand gesture library API and issues.
    - Details on the camera movement detection and math related debugging for movement.