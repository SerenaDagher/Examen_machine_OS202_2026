# Simulation d'une galaxie à n corps en utilisant une grille spatiale pour accélérer le calcul des forces gravitationnelles.
#     On crée une classe représentant le système de corps avec la méthode d'intégration basée sur une grille.
# On utilise numba pour accélérer les calculs.
import numpy as np
import visualizer3d
import sys
from numba import njit, prange
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Unités:
# - Distance: année-lumière (ly)
# - Masse: masse solaire (M_sun)
# - Vitesse: année-lumière par an (ly/an)
# - Temps: année

# Constante gravitationnelle en unités [ly^3 / (M_sun * an^2)]
G = 1.560339e-13

def generate_star_color(mass : float) -> tuple[int, int, int]:
    """
    Génère une couleur pour une étoile en fonction de sa masse.
    Les étoiles massives sont bleues, les moyennes sont jaunes, les petites sont rouges.
    
    Parameters:
    -----------
    mass : float
        Masse de l'étoile en masses solaires
    
    Returns:
    --------
    color : tuple
        Couleur RGB (R, G, B) avec des valeurs entre 0 et 255
    """
    if mass > 5.0:
        return (150, 180, 255)
    elif mass > 2.0:
        return (255, 255, 255)
    elif mass >= 1.0:
        return (255, 255, 200)
    else:
        return (255, 150, 100)

@njit
def update_stars_in_grid(cell_start_indices : np.ndarray, body_indices : np.ndarray,
                         cell_masses : np.ndarray, cell_com_positions : np.ndarray,
                         masses: np.ndarray,
                         positions : np.ndarray, grid_min : np.ndarray, grid_max : np.ndarray,
                         cell_size : np.ndarray, n_cells : np.ndarray):
    n_bodies = positions.shape[0]
    cell_start_indices.fill(-1)
    cell_counts = np.zeros(shape=(np.prod(n_cells),), dtype=np.int64)

    for ibody in range(n_bodies):
        cell_idx = np.floor((positions[ibody] - grid_min) / cell_size).astype(np.int64)
        for i in range(3):
            if cell_idx[i] >= n_cells[i]:
                cell_idx[i] = n_cells[i] - 1
            elif cell_idx[i] < 0:
                cell_idx[i] = 0
        morse_idx = cell_idx[0] + cell_idx[1]*n_cells[0] + cell_idx[2]*n_cells[0]*n_cells[1]
        cell_counts[morse_idx] += 1

    running_index = 0
    for i in range(len(cell_counts)):
        cell_start_indices[i] = running_index
        running_index += cell_counts[i]
    cell_start_indices[len(cell_counts)] = running_index

    current_counts = np.zeros(shape=(np.prod(n_cells),), dtype=np.int64)
    for ibody in range(n_bodies):
        cell_idx = np.floor((positions[ibody] - grid_min) / cell_size).astype(np.int64)
        for i in range(3):
            if cell_idx[i] >= n_cells[i]:
                cell_idx[i] = n_cells[i] - 1
            elif cell_idx[i] < 0:
                cell_idx[i] = 0
        morse_idx = cell_idx[0] + cell_idx[1]*n_cells[0] + cell_idx[2]*n_cells[0]*n_cells[1]
        index_in_cell = cell_start_indices[morse_idx] + current_counts[morse_idx]
        body_indices[index_in_cell] = ibody
        current_counts[morse_idx] += 1

    for i in prange(len(cell_counts)):
        cell_mass = 0.0
        com_position = np.zeros(3, dtype=np.float32)
        start_idx = cell_start_indices[i]
        end_idx = cell_start_indices[i+1]
        for j in range(start_idx, end_idx):
            ibody = body_indices[j]
            m = masses[ibody]
            cell_mass += m
            com_position += positions[ibody] * m
        if cell_mass > 0.0:
            com_position /= cell_mass
        cell_masses[i] = cell_mass
        cell_com_positions[i] = com_position

@njit(parallel=True)
def compute_acceleration_local(positions : np.ndarray, masses : np.ndarray,
                               body_global_ids : np.ndarray,
                               cell_start_indices : np.ndarray, body_indices : np.ndarray,
                               cell_masses : np.ndarray, cell_com_positions : np.ndarray,
                               grid_min : np.ndarray,
                               cell_size : np.ndarray, n_cells : np.ndarray):
    n_bodies = body_global_ids.shape[0]
    a = np.zeros((n_bodies, 3), dtype=np.float32)
    for ibody in prange(n_bodies):
        global_id = body_global_ids[ibody]
        pos = positions[global_id]
        cell_idx = np.floor((pos - grid_min) / cell_size).astype(np.int64)

        for i in range(3):
            if cell_idx[i] >= n_cells[i]:
                cell_idx[i] = n_cells[i] - 1
            elif cell_idx[i] < 0:
                cell_idx[i] = 0

        for ix in range(n_cells[0]):
            for iy in range(n_cells[1]):
                for iz in range(n_cells[2]):
                    morse_idx = ix + iy*n_cells[0] + iz*n_cells[0]*n_cells[1]
                    if (abs(ix-cell_idx[0]) > 2) or (abs(iy-cell_idx[1]) > 2) or (abs(iz-cell_idx[2]) > 2):
                        cell_mass = cell_masses[morse_idx]
                        if cell_mass > 0.0:
                            direction = cell_com_positions[morse_idx] - pos
                            distance = np.sqrt(direction[0]**2 + direction[1]**2 + direction[2]**2)
                            if distance > 1.E-10:
                                inv_dist3 = 1.0 / (distance ** 3)
                                a[ibody,:] += G * direction[:] * inv_dist3 * cell_mass
                    else:
                        start_idx = cell_start_indices[morse_idx]
                        end_idx = cell_start_indices[morse_idx+1]
                        for j in range(start_idx, end_idx):
                            jbody = body_indices[j]
                            if jbody != global_id:
                                direction = positions[jbody] - pos
                                distance = np.sqrt(direction[0]**2 + direction[1]**2 + direction[2]**2)
                                if distance > 1.E-10:
                                    inv_dist3 = 1.0 / (distance ** 3)
                                    a[ibody,:] += G * direction[:] * inv_dist3 * masses[jbody]
    return a

class SpatialGrid:
    def __init__(self, positions : np.ndarray, nb_cells_per_dim : tuple[int, int, int]):
        self.min_bounds = np.min(positions, axis=0) - 1.E-6
        self.max_bounds = np.max(positions, axis=0) + 1.E-6
        self.n_cells = np.array(nb_cells_per_dim)
        self.cell_size = (self.max_bounds - self.min_bounds) / self.n_cells
        self.cell_start_indices = np.full(np.prod(self.n_cells) + 1, -1, dtype=np.int64)
        self.body_indices = np.empty(shape=(positions.shape[0],), dtype=np.int64)
        self.cell_masses = np.zeros(shape=(np.prod(self.n_cells),), dtype=np.float32)
        self.cell_com_positions = np.zeros(shape=(np.prod(self.n_cells), 3), dtype=np.float32)

    def update(self, positions : np.ndarray, masses : np.ndarray):
        update_stars_in_grid(self.cell_start_indices, self.body_indices,
                             self.cell_masses, self.cell_com_positions,
                             masses,
                             positions, self.min_bounds, self.max_bounds,
                             self.cell_size, self.n_cells)

class NBodySystem:
    def __init__(self, filename, ncells_per_dir : tuple[int, int, int] = (20,20,1)):
        positions = []
        velocities = []
        masses = []

        self.max_mass = 0.
        self.box = np.array([[-1.E-6,-1.E-6,-1.E-6],[1.E-6,1.E-6,1.E-6]], dtype=np.float64)

        with open(filename, "r") as fich:
            line = fich.readline()
            while line:
                data = line.split()
                masses.append(float(data[0]))
                positions.append([float(data[1]), float(data[2]), float(data[3])])
                velocities.append([float(data[4]), float(data[5]), float(data[6])])
                self.max_mass = max(self.max_mass, masses[-1])

                for i in range(3):
                    self.box[0][i] = min(self.box[0][i], positions[-1][i]-1.E-6)
                    self.box[1][i] = max(self.box[1][i], positions[-1][i]+1.E-6)

                line = fich.readline()

        self.all_positions = np.array(positions, dtype=np.float32)
        self.all_velocities = np.array(velocities, dtype=np.float32)
        self.all_masses = np.array(masses, dtype=np.float32)
        self.colors = [generate_star_color(m) for m in masses]

        self.grid = SpatialGrid(self.all_positions, ncells_per_dir)
        self.n_cells = self.grid.n_cells
        self.grid_min = self.grid.min_bounds
        self.grid_max = self.grid.max_bounds
        self.cell_size = self.grid.cell_size

        self.local_ids = self.compute_local_ids(self.all_positions)
        self.extract_local_arrays()

    def owner_of_position(self, x):
        x0 = self.grid_min[0]
        x1 = self.grid_max[0]
        width = (x1 - x0) / size
        r = int((x - x0) / width)
        if r < 0:
            r = 0
        if r >= size:
            r = size - 1
        return r

    def compute_local_ids(self, positions):
        ids = []
        for i in range(positions.shape[0]):
            if self.owner_of_position(positions[i,0]) == rank:
                ids.append(i)
        return np.array(ids, dtype=np.int64)

    def extract_local_arrays(self):
        self.positions = self.all_positions[self.local_ids].copy()
        self.velocities = self.all_velocities[self.local_ids].copy()
        self.masses = self.all_masses[self.local_ids].copy()

    def rebuild_global_arrays(self):
        local_count = np.array([self.positions.shape[0]], dtype=np.int64)
        counts = np.empty(size, dtype=np.int64)
        comm.Allgather(local_count, counts)

        send_pos = self.positions.astype(np.float32).reshape(-1)
        send_vel = self.velocities.astype(np.float32).reshape(-1)
        send_mass = self.masses.astype(np.float32)

        recv_counts_pos = (counts * 3).astype(np.int64)
        recv_counts_mass = counts.copy()

        displs_pos = np.zeros(size, dtype=np.int64)
        displs_mass = np.zeros(size, dtype=np.int64)
        for i in range(1, size):
            displs_pos[i] = displs_pos[i-1] + recv_counts_pos[i-1]
            displs_mass[i] = displs_mass[i-1] + recv_counts_mass[i-1]

        recv_pos = np.empty(np.sum(recv_counts_pos), dtype=np.float32)
        recv_vel = np.empty(np.sum(recv_counts_pos), dtype=np.float32)
        recv_mass = np.empty(np.sum(recv_counts_mass), dtype=np.float32)

        comm.Allgatherv(send_pos, [recv_pos, recv_counts_pos, displs_pos, MPI.FLOAT])
        comm.Allgatherv(send_vel, [recv_vel, recv_counts_pos, displs_pos, MPI.FLOAT])
        comm.Allgatherv(send_mass, [recv_mass, recv_counts_mass, displs_mass, MPI.FLOAT])

        self.all_positions = recv_pos.reshape(-1, 3)
        self.all_velocities = recv_vel.reshape(-1, 3)
        self.all_masses = recv_mass

    def compute_global_cells(self):
        self.grid.update(self.all_positions, self.all_masses)

        local_mass = self.grid.cell_masses.copy()
        local_com_mass = self.grid.cell_com_positions * local_mass[:, None]

        global_mass = np.zeros_like(local_mass)
        global_com_mass = np.zeros_like(local_com_mass)

        comm.Allreduce(local_mass, global_mass, op=MPI.SUM)
        comm.Allreduce(local_com_mass, global_com_mass, op=MPI.SUM)

        self.grid.cell_masses[:] = global_mass[:]
        self.grid.cell_com_positions[:] = 0.0
        nz = global_mass > 0.0
        self.grid.cell_com_positions[nz] = global_com_mass[nz] / global_mass[nz, None]

    def redistribute_after_move(self):
        self.rebuild_global_arrays()
        self.local_ids = self.compute_local_ids(self.all_positions)
        self.extract_local_arrays()

    def update_positions(self, dt):
        self.rebuild_global_arrays()
        self.compute_global_cells()

        a = compute_acceleration_local(
            self.all_positions, self.all_masses, self.local_ids,
            self.grid.cell_start_indices, self.grid.body_indices,
            self.grid.cell_masses, self.grid.cell_com_positions,
            self.grid.min_bounds,
            self.grid.cell_size, self.grid.n_cells
        )

        self.positions += self.velocities * dt + 0.5 * a * dt * dt

        self.rebuild_global_arrays()
        self.compute_global_cells()

        a_new = compute_acceleration_local(
            self.all_positions, self.all_masses, self.local_ids,
            self.grid.cell_start_indices, self.grid.body_indices,
            self.grid.cell_masses, self.grid.cell_com_positions,
            self.grid.min_bounds,
            self.grid.cell_size, self.grid.n_cells
        )

        self.velocities += 0.5 * (a + a_new) * dt
        self.redistribute_after_move()

system : NBodySystem

def gather_positions_root(local_positions):
    local_count = np.array([local_positions.shape[0]], dtype=np.int64)
    counts = np.empty(size, dtype=np.int64)
    comm.Allgather(local_count, counts)

    sendbuf = local_positions.astype(np.float32).reshape(-1)
    recv_counts = (counts * 3).astype(np.int64)
    displs = np.zeros(size, dtype=np.int64)
    for i in range(1, size):
        displs[i] = displs[i-1] + recv_counts[i-1]

    recvbuf = None
    if rank == 0:
        recvbuf = np.empty(np.sum(recv_counts), dtype=np.float32)

    comm.Gatherv(sendbuf, [recvbuf, recv_counts, displs, MPI.FLOAT], root=0)

    if rank == 0:
        return recvbuf.reshape(-1, 3)
    return None

def update_positions(dt : float):
    global system
    import time

    t0 = time.time()
    system.update_positions(dt)
    t1 = time.time()

    if rank == 0:
        print(f"Update time: {(t1 - t0)*1000:.2f} ms")

    gathered = gather_positions_root(system.positions)
    if rank == 0:
        return gathered
    return None

def worker_loop(dt : float):
    global system
    while True:
        system.update_positions(dt)
        gather_positions_root(system.positions)

def run_simulation(filename, geometry=(800,600), ncells_per_dir : tuple[int, int, int] = (20,20,1), dt=0.001):
    global system
    system = NBodySystem(filename, ncells_per_dir=ncells_per_dir)

    if rank != 0:
        worker_loop(dt)
        return

    pos = system.all_positions
    col = system.colors
    intensity = np.clip(system.all_masses / system.max_mass, 0.5, 1.0)
    visu = visualizer3d.Visualizer3D(
        pos, col, intensity,
        [[system.box[0][0], system.box[1][0]],
         [system.box[0][1], system.box[1][1]],
         [system.box[0][2], system.box[1][2]]]
    )
    visu.run(updater=update_positions, dt=dt)

filename = "data/galaxy_1000"
dt = 0.001
n_cells_per_dir = (20,20,1)

if len(sys.argv) > 1:
    filename = sys.argv[1]
if len(sys.argv) > 2:
    dt = float(sys.argv[2])
if len(sys.argv) > 5:
    n_cells_per_dir = (int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))

if rank == 0:
    print(f"Simulation de {filename} avec dt = {dt} et grille {n_cells_per_dir} sur {size} processus MPI")

run_simulation(filename, ncells_per_dir=n_cells_per_dir, dt=dt)