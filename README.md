# State-Consistency Loss for Learning Spatial Perception Tasks from Partial Labels

*Mirko Nava, Luca Maria Gambardella, and Alessandro Giusti*

Dalle Molle Institute for Artificial Intelligence, USI-SUPSI, Lugano (Switzerland)

### Abstract

When learning models for real-world robot spatial perception tasks, one might have access only to partial labels: this occurs for example in semi-supervised scenarios (in which labels are not available for a subset of the training instances) or in some types of self-supervised robot learning (where the robot autonomously acquires a labeled training set, but only acquires labels for a subset of the output variables in each instance).  We introduce a general approach to deal with this class of problems using an auxiliary loss enforcing the expectation that the perceived environment state should not abruptly change; then, we instantiate the approach to solve two robot perception problems: a simulated ground robot learning long-range obstacle mapping as a 400-binary-label classification task in a self-supervised way in a static environment; and a real nano-quadrotor learning human pose estimation as a 3-variable regression task in a semi-supervised way in a dynamic environment.  In both cases, our approach yields significant quantitative performance improvements (average increase of 6 AUC percentage points in the former; relative improvement of the R2 metric ranging from 7% to 33% in the latter) over baselines.

![Predictions](https://github.com/idsia-robotics/state-consistency-loss/blob/main/img/occupancy_map.png "Predictions")
*Self-supervised occupancy map estimation: on six testing instances. **Left**: input before down-scaling.  **Center**: self-supervised labels (not used for prediction).  **Right**: model prediction and FOV (blue). Red represents occupied cells, green for empty cells, and gray for missing information.*

![Predictions](https://github.com/idsia-robotics/state-consistency-loss/blob/main/img/user_pose.png "Predictions")
*Semi-supervised estimation of user pose in a nano-drone: user's head location (x, y, z) and heading (phi) predictions.
Compared to the model trained using only the task loss (blue), our approach (red) is usually closer to the ground-truth (green).*

The PDF of the article is available in Open Access [here](https://doi.org/10.1109/LRA.2021.3056378).

### Bibtex

```properties
@article{nava2021ral,
  author={M. {Nava} and L. M. {Gambardella} and A. {Giusti}},
  journal={IEEE Robotics and Automation Letters}, 
  title={State-Consistency Loss for Learning Spatial Perception Tasks From Partial Labels}, 
  year={2021},
  volume={6},
  number={2},
  pages={1112-1119},
  doi={10.1109/LRA.2021.3056378}
}
```

### Videos

[![Learning Long-Range Perception Using Self-Supervision from Short-Range Sensors and Odometry](https://github.com/idsia-robotics/state-consistency-loss/blob/main/video/video.gif)](https://youtu.be/AD69cYFinzc)

All the video material of models trained with the proposed approach on different scenarios, robots and systems is available [here](https://github.com/idsia-robotics/state-consistency-loss/tree/main/video).

### Code

The entire codebase is avaliable [here](https://github.com/idsia-robotics/state-consistency-loss/tree/main/code).
In order to generate the datasets one should launch the script preprocess.py which will create the dataset in hdf5 file format, starting from a collection of ROS bagfiles stored in a given folder.

The script train.py is used to train the model, which is defined in model.py, using a given hdf5 dataset. A list of the available parameters can be seen by launching  `python train.py -h `.

The script test.py is used to test the model using a subset of the hdf5 groups defined in the script. A list of the available parameters can be seen by launching  `python test.py -h `.
