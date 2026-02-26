from logging import info
import os
import re
from typing import Dict


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
        self.events: Dict[int, Dict[str, int]] = {}
        self.scan()

    def scan(self):
        """
        Scan the base directory and build the event dictionary.
        """
        self.events.clear()
        event_id = 0

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
            
            self.events[event_id] = {
                "campaignID": campaign_id,
                "clusterID": cluster_id,
                "jobID": job_id,
                "UID": unique_id,
                "ran": is_run_complete,
                "folder_path": entry.path,
                "h5_path": self.find_h5_file(entry.path+"/MUSIC")
            }
          
            event_id += 1
    
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
            raise FileNotFoundError(f"No .h5 file found in: {folder_path}")
        if len(h5_files) > 1:
            raise ValueError(f"Expected exactly one .h5 file, but found {len(h5_files)}: {h5_files}")

        return h5_files[0]

    def get_all_h5_paths(self):
        """
        Get all h5 paths as a list.
        """
        return [self.events[i]["h5_path"] for i in self.events.keys()]
