# ImSim Analysis Scripts

## Usage


### correlation.py
* Takes "track" t,X,Y csv file as generated by the main_interface.py
script in the FishVideo project (https://github.com/msmith3-uottawa-ca/FishVideo)
* Takes "profile" t,R[1-10] csv file as accepted by the Fish_Gui project
(https://github.com/aaronshifman/Fish_Gui) (Currently not used)
* Takes "asrun" t,R[1-10] .npy file as generated by the Fish_Gui project
(https://github.com/aaronshifman/Fish_Gui)
* Takes a startoffset in seconds. This is the time the light flashes in the video,
indicating the beginning of an experimental run.
* Run in interactive console. Aligned data is found in variable corlist
* Function "bootstrap" shuffles the order of the fish X position for statistical bootstrapping. 
Operates in place! 