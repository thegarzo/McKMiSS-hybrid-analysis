import numpy as np

def read_iSS_binary(filename):
    events = []
    
    particle_dtype = np.dtype([
        ('pid',  np.int32),
        ('mass', np.float32),
        ('t',    np.float32),
        ('x',    np.float32),
        ('y',    np.float32),
        ('z',    np.float32),
        ('E',    np.float32),
        ('px',   np.float32),
        ('py',   np.float32),
        ('pz',   np.float32),
    ])
    
    with open(filename, 'rb') as f:
        while True:
            # read number of particles in this event
            raw = f.read(4)
            if not raw or len(raw) < 4:
                break
            
            n_particles = np.frombuffer(raw, dtype=np.int32)[0]
            
            if n_particles == 0:
                events.append(np.array([], dtype=particle_dtype))
                continue
            
            raw = f.read(n_particles * particle_dtype.itemsize)
            if len(raw) < n_particles * particle_dtype.itemsize:
                break
            
            particles = np.frombuffer(raw, dtype=particle_dtype)
            events.append(particles)
    
    return events
