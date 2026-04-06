from logging import info
import os
import re
from typing import Dict

PHYSICS_EMPTY_PHRASE = "No freeze-out fluid cell, exit now"

class Parser:
    """
    Index folders named:
        out_<campaignID>_<clusterID>_<jobID>

    Produces a dictionary:
        event_id -> {campaignID, clusterID, jobID}
    """

    FOLDER_PATTERN = re.compile(r"^out_(\d+)_(\d+)_(\d+)$")

    def __init__(self, base_path: str):
        self.base_path = base_path
        self.logs_path = os.path.join(os.path.dirname(base_path), "logs")
        # print(self.logs_path/)
        self.events: Dict[int, Dict[str, int]] = {}
        self.events: Dict[int, Dict[str, int]] = {}
        self.scan()
        
    def _is_physics_empty(self, cluster_id: int, job_id: int) -> bool:
        """ 
        Return True if the .out log for this job contains the freeze-out phrase,
        indicating the system was too cold to produce particles (expected physics).
        Returns False if the log is missing or the phrase is absent.
        """ 
        log_file = os.path.join(self.logs_path,f"run_{cluster_id}_{job_id}.out")
        if not os.path.isfile(log_file):
            return False 
        try:
            with open(log_file, "r", errors="replace") as f:
                return PHYSICS_EMPTY_PHRASE in f.read()
        except OSError:
            return False
         
    def scan(self):
        """
        Scan the base directory and build the event dictionary.
        """
        self.events.clear()
        event_id = 0

        skipped =0
        nonsense =0
        for entry in os.scandir(self.base_path):
            if not entry.is_dir():
                continue

            match = self.FOLDER_PATTERN.match(entry.name)
            if not match:
                continue

            campaign_id, cluster_id, job_id = map(int, match.groups())
            unique_id = str(cluster_id)+str(job_id)

            
            subfolder_path = os.path.join(entry.path, "MUSIC", "outputs")
            is_run_complete= os.path.isdir(subfolder_path)

            h5_path =self.find_h5_file(entry.path+"/MUSIC")

            if h5_path is None:
                if self._is_physics_empty(cluster_id, job_id):
                    self.events[event_id] = {
                        "campaignID": campaign_id,
                        "clusterID": cluster_id,
                        "jobID": job_id,
                        "UID": unique_id,
                        "ran": is_run_complete,
                        "folder_path": entry.path,
                        "h5_path": "empty_event"
                    }
                    skipped += 1
                    event_id += 1
                else: 
                    print(entry.path)
                    nonsense+=1
            else:
                self.events[event_id] = {
                    "campaignID": campaign_id,
                    "clusterID": cluster_id,
                    "jobID": job_id,
                    "UID": unique_id,
                    "ran": is_run_complete,
                    "folder_path": entry.path,
                    "h5_path": h5_path
                }
                event_id += 1
        
        print("Failed events are  n = " +str(nonsense)+  ' files')
        print("Empty events n = " +str(skipped)+  ' files')
        print("Catalogued n = " +str(event_id)+  ' files')
        print("Empty events are n = " +f'{100*float(skipped)/float(event_id):.2f}' +  ' percent of files')
    
    def get_event_folder(self, event_id: int):
        """
        Get the folder path for a given event ID.
        """
        if event_id not in self.events:
            raise ValueError(f"Event ID {event_id} not found.")
        
        return self.events[event_id]["path"]
    
    def get_all_events(self):
        """
        Get all folder paths as a list.
        """
        return [self.events[i]["folder_path"] for i in self.events.keys()]

    def find_h5_file(self,folder_path):
        """
        Locates the single .h5 file in the given folder and returns its full path.

        Args:
            folder_path: Path to the folder to search in.

        Returns:
            Full path to the .h5 file.

        Raises:
            FileNotFoundError: If no .h5 file is found.
            ValueError: If more than one .h5 file is found.
        """
        h5_files = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.endswith(".h5")
        ]

        if len(h5_files) == 0:
            return None
        if len(h5_files) > 1:
            raise ValueError(f"Expected exactly one .h5 file, but found {len(h5_files)}: {h5_files}")

        return h5_files[0]

    def get_all_h5_paths(self):
        """
        Get all h5 paths as a list.
        """
        return [self.events[i]["h5_path"] for i in self.events.keys()]
