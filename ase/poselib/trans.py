import numpy as np
data = np.load("data/smplh.npy", allow_pickle = True).item()

data['rotation']['arr'] = data['rotation']['arr'][0] 
data['root_translation']['arr'] = data['root_translation']['arr'][0] 
data['__name__'] = 'SkeletonState'
data.pop('global_velocity')
data.pop('global_angular_velocity')
data.pop('fps')
data['root_translation']['arr'] = np.array([0,0,0])
for i in range(data['rotation']['arr'].shape[0]):
    data['rotation']['arr'][i] = np.array([0,0,0,1])

np.save("data/smplh_avg_tpose.npy", data)