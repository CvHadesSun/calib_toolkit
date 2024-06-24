# calib toolkit box


## env 
```
open3d
opencv-python
prettytable
colorama

```


## params list
```
-s --src_path : the calib camera data root dir
-o --output_path : the calib camera data root dir
-b --chessboard : the chessboard size : (rows,cols),default=(11, 8)
-l --square_length : the chessboard square size (mm), default=20
-n --cam_num : the number of needed to calib
-t --thread_num : the multithred pool size, default=5
-v --verbose : Enable verbose output
--with_refine : with 3d pts refine processing
--undistort : undistort the rgb and depth
--operation : uthe calib operation: (0: calib), (1: error inference), (2: pcd vis)
--frame_id : the pcd vis frame id,default=0
--cam_ids : vis which camera pcd, defalut = [all cameras.]
--vis_pcd : vis raw depth camera pcds,default=[]

```


## pipeline
1. calib camera

```
python demo.py -s path/to/data --operation 0 -n 16 
```

2. vis cpd
```
python demo.py -s path/to/data --operation 2 -n 16 --frame_id 10
```

3. compute error.
```
python demo.py -s path/to/data --operation 1 -n 16 
```

4. compute error.
```
python demo.py -s path/to/data --operation 3 -n 16 --vis_pcd Cam0 Cam1 Cam2 Cam10 --frame_id 0
```