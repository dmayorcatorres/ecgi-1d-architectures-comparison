from mpl_toolkits.mplot3d.art3d import Poly3DCollection
###############################################################################
# CLASE ECGiVisualizer (visualización)
###############################################################################
class ECGiVisualizer:
    """
    Clase encargada de las rutinas de visualización de señales (ECG/EMG) y sus
    transformadas (CWT, DWT, etc.).
    """

    @staticmethod
    def plot_electrodes(
        original_data_3d: Union[np.ndarray, torch.Tensor],
        fs: int = 1000,
        signal_label: str = 'ECG',
        signal_index: int = 0,
        electrode_index: Optional[int] = None,
        num_to_plot: Optional[int] = None,
        reconstructed_data_3d: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> None:
        """
        Grafica señales de un tensor 3D (n_signals, n_time, n_electrodes).

        Dos modos de uso:
          1) Si 'electrode_index' no es None => se dibuja un solo electrodo (azul).
             Y si 'reconstructed_data_3d' se pasa, se dibuja encima en rojo.
          2) Si 'electrode_index' es None y 'num_to_plot' no es None => submuestreo
             de electrodos en subplots.
        """
        # Si es torch.Tensor, mover a CPU y a NumPy
        if isinstance(original_data_3d, torch.Tensor):
            original_data_3d = original_data_3d.cpu().numpy()
        if isinstance(reconstructed_data_3d, torch.Tensor):
            reconstructed_data_3d = reconstructed_data_3d.cpu().numpy()

        n_signals, n_time, n_electrodes = original_data_3d.shape
        if not (0 <= signal_index < n_signals):
            print(f"signal_index fuera de rango: 0..{n_signals-1}")
            return

        has_reconstruction = False
        if reconstructed_data_3d is not None:
            rec_ns, rec_nt, rec_ne = reconstructed_data_3d.shape
            if rec_ns == n_signals and rec_nt == n_time and rec_ne == n_electrodes:
                has_reconstruction = True
            else:
                print("Shape mismatch en 'reconstructed_data_3d'. Se ignora reconstrucción.")
                reconstructed_data_3d = None

        # MODO 1: single electrode
        if electrode_index is not None:
            if not (0 <= electrode_index < n_electrodes):
                print(f"electrode_index fuera de rango: 0..{n_electrodes-1}")
                return

            original_series = original_data_3d[signal_index, :, electrode_index]
            time_axis = np.arange(n_time) / fs

            plt.figure(figsize=(10, 4))
            plt.plot(time_axis, original_series, color='blue', label='Original')

            if has_reconstruction:
                rec_series = reconstructed_data_3d[signal_index, :, electrode_index]
                plt.plot(time_axis, rec_series, color='red', label='Reconstructed')

            plt.title(f'{signal_label} - Signal {signal_index}, Electrode {electrode_index}')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            plt.legend()
            plt.grid(True)
            plt.show()
            return

        # MODO 2: submuestreo
        if num_to_plot is not None:
            if num_to_plot <= 0:
                print("num_to_plot debe ser > 0.")
                return

            signal_2d = original_data_3d[signal_index]  # (n_time, n_electrodes)
            step = max(1, n_electrodes // num_to_plot)
            selected_electrodes = list(range(0, n_electrodes, step))[:num_to_plot]

            time_axis = np.arange(n_time) / fs
            num_cols = 4
            num_rows = (num_to_plot + num_cols - 1) // num_cols

            plt.figure(figsize=(14, num_rows * 3))

            for i, e_idx in enumerate(selected_electrodes):
                plt.subplot(num_rows, num_cols, i + 1)
                plt.plot(time_axis, signal_2d[:, e_idx], color='blue', label='Original')

                if has_reconstruction:
                    rec_2d = reconstructed_data_3d[signal_index]  # (n_time, n_electrodes)
                    plt.plot(time_axis, rec_2d[:, e_idx], color='red', label='Reconstructed')

                plt.title(f'Electrode {e_idx}')
                plt.xlabel('Time (s)')
                plt.ylabel('Amplitude')
                plt.legend()
                plt.tight_layout()

            plt.suptitle(
                f'{signal_label} - Signal {signal_index} (Subsampling {num_to_plot} electrodes)',
                fontsize=16, y=1.02
            )
            plt.show()
            return

        print("Debes especificar 'electrode_index' (single electrode) o 'num_to_plot' (subsample).")

    @staticmethod
    def plot_cwt(
        cwt_tensor_4d: Union[np.ndarray, torch.Tensor],
        signal_index: int = 0,
        electrode_index: int = 0,
        wavelet_label: str = 'CWT'
    ) -> None:
        """
        Dibuja la CWT 2D (scales x time) para un 'signal_index' y 'electrode_index'.
        Se asume que cwt_tensor_4d es (n_signals, n_scales, n_time, n_electrodes).
        """
        # Convertir a NumPy si es Torch
        if isinstance(cwt_tensor_4d, torch.Tensor):
            cwt_tensor_4d = cwt_tensor_4d.cpu().numpy()

        n_signals, n_scales, n_time, n_electrodes = cwt_tensor_4d.shape
        if not (0 <= signal_index < n_signals):
            print(f"signal_index fuera de rango: 0..{n_signals-1}")
            return
        if not (0 <= electrode_index < n_electrodes):
            print(f"electrode_index fuera de rango: 0..{n_electrodes-1}")
            return

        cwt_2d = cwt_tensor_4d[signal_index, :, :, electrode_index]  # (n_scales, n_time)

        plt.figure(figsize=(10, 6))
        plt.imshow(cwt_2d, aspect='auto', cmap='viridis', origin='lower')
        plt.colorbar(label='Amplitude')
        plt.title(f'{wavelet_label} - Signal={signal_index}, Electrode={electrode_index}')
        plt.xlabel('Time axis')
        plt.ylabel('Scales')
        plt.show()

    @staticmethod
    def plot_dwt(
        dwt_tensor_4d: Union[np.ndarray, torch.Tensor],
        signal_index: int = 0,
        electrode_index: int = 0,
        wavelet_label: str = 'DWT'
    ) -> None:
        """
        Dibuja la DWT 2D (levels x time) para un 'signal_index' y 'electrode_index'.
        Se asume que dwt_tensor_4d es (n_signals, n_levels, n_time, n_electrodes).
        """
        if isinstance(dwt_tensor_4d, torch.Tensor):
            dwt_tensor_4d = dwt_tensor_4d.cpu().numpy()

        n_signals, n_levels, n_time, n_electrodes = dwt_tensor_4d.shape
        if not (0 <= signal_index < n_signals):
            print(f"signal_index fuera de rango: 0..{n_signals-1}")
            return
        if not (0 <= electrode_index < n_electrodes):
            print(f"electrode_index fuera de rango: 0..{n_electrodes-1}")
            return

        dwt_2d = dwt_tensor_4d[signal_index, :, :, electrode_index]  # (n_levels, n_time)

        plt.figure(figsize=(10, 6))
        plt.imshow(dwt_2d, aspect='auto', cmap='viridis', origin='lower')
        plt.colorbar(label='Amplitude')
        plt.title(f'{wavelet_label} - Signal={signal_index}, Electrode={electrode_index}')
        plt.xlabel('Time axis')
        plt.ylabel('Levels')
        plt.show()

    @staticmethod
    def plot_structure_mesh(
        nodes: Union[np.ndarray, List[List[float]]],
        faces: Union[np.ndarray, List[List[int]]],
        structure_label: str = "Structure"
    ) -> None:
        """
        Visualiza en 3D la malla de una estructura usando los nodos y la conectividad de caras.

        Args:
            nodes: Array de nodos. Se asume que puede venir en forma (3, n) (cada columna es [x,y,z])
                   o (n, 3). Si es (3, n), se transpone a (n, 3).
            faces: Array de conectividad. Se asume que puede venir en forma (3, m) (cada columna contiene
                   los índices de los nodos de un triángulo) o (m, 3). Se transpone a (m, 3) si es necesario.
            structure_label: Etiqueta para el título (ejemplo: "Heart", "Torso").
        """
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        try:
            nodes = np.array(nodes)
            faces = np.array(faces, dtype=int)
        except Exception as e:
            logger.error(f"Error processing nodes or faces: {e}")
            return

        # Si nodes viene en forma (3, n), transpónalo a (n, 3)
        if nodes.shape[0] == 3 and nodes.shape[1] > 3:
            nodes = nodes.T

        # Si faces viene en forma (3, m), transpónelo a (m, 3)
        if faces.shape[0] == 3 and faces.shape[1] > 3:
            faces = faces.T

        # Ajustar índices si son 1-indexados (MATLAB)
        if np.min(faces) == 1:
            faces = faces - 1

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Crear la lista de polígonos usando los índices de faces en nodes
        polys = [nodes[face] for face in faces]
        mesh = Poly3DCollection(polys, alpha=0.5, edgecolor='k', linewidths=0.5)
        mesh.set_facecolor('cyan')
        ax.add_collection3d(mesh)

        # Graficar los nodos como puntos rojos
        ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], color='red', s=10)

        # Ajustar límites
        margin = 0.1
        ax.set_xlim(np.min(nodes[:, 0]) - margin, np.max(nodes[:, 0]) + margin)
        ax.set_ylim(np.min(nodes[:, 1]) - margin, np.max(nodes[:, 1]) + margin)
        ax.set_zlim(np.min(nodes[:, 2]) - margin, np.max(nodes[:, 2]) + margin)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"{structure_label} Mesh")
        plt.tight_layout()
        plt.show()


    @staticmethod
    def plot_structure_mesh(
        nodes: Union[np.ndarray, List[List[float]]],
        faces: Union[np.ndarray, List[List[int]]],
        structure_label: str = "Structure",
        cmap: str = 'viridis'
    ) -> None:
        """
        Visualiza en 3D la malla de una estructura usando los nodos y la conectividad de caras.
        Cada nodo se muestra como un punto, y las caras se dibujan como polígonos con líneas delgadas.

        Args:
            nodes: Array de nodos. Se asume que puede venir en forma (3, n) (cada columna es [x,y,z])
                  o (n, 3). Si es (3, n), se transpone a (n, 3).
            faces: Array de conectividad. Se asume que puede venir en forma (3, m) (cada columna contiene
                  los índices de los nodos de un triángulo) o (m, 3). Se transpone a (m, 3) si es necesario.
            structure_label: Etiqueta para el título (ejemplo: "Heart", "Torso").
            cmap: Mapa de colores (no usado en este caso, pero puede extenderse para rellenos).
        """
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        import matplotlib.pyplot as plt

        try:
            nodes = np.array(nodes)
            faces = np.array(faces, dtype=int)
        except Exception as e:
            logger.error(f"Error processing nodes or faces: {e}")
            return

        # Si nodes viene en forma (3, n), transpónalo a (n, 3)
        if nodes.shape[0] == 3 and nodes.shape[1] > 3:
            nodes = nodes.T

        # Si faces viene en forma (3, m), transpónelas a (m, 3)
        if faces.shape[0] == 3 and faces.shape[1] > 3:
            faces = faces.T

        # Ajustar índices si son 1-indexados (MATLAB)
        if np.min(faces) == 1:
            faces = faces - 1

        # Construir la lista de polígonos, validando que cada cara tenga índices válidos
        polys = []
        for face in faces:
            if np.any(face < 0) or np.any(face >= nodes.shape[0]):
                logger.warning(f"Se omite la cara con índices fuera de rango: {face}")
                continue
            polys.append(nodes[face])

        if len(polys) == 0:
            logger.error("No se han encontrado caras válidas para la visualización.")
            return

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        mesh = Poly3DCollection(polys, alpha=0.5, edgecolor='k', linewidths=0.5)
        mesh.set_facecolor('cyan')
        ax.add_collection3d(mesh)

        # Graficar los nodos como puntos rojos
        ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], color='red', s=10)

        margin = 0.1
        ax.set_xlim(np.min(nodes[:, 0]) - margin, np.max(nodes[:, 0]) + margin)
        ax.set_ylim(np.min(nodes[:, 1]) - margin, np.max(nodes[:, 1]) + margin)
        ax.set_zlim(np.min(nodes[:, 2]) - margin, np.max(nodes[:, 2]) + margin)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"{structure_label} Mesh")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_mesh_with_pseudocolor(
        nodes: Union[np.ndarray, List[List[float]]],
        faces: Union[np.ndarray, List[List[int]]],
        potentials: Union[np.ndarray, List[float]],
        structure_label: str = "Structure",
        cmap: str = 'viridis'
    ) -> None:
        """
        Visualiza en 3D la malla de una estructura y colorea cada nodo según el potencial.

        Se asume que:
          - Los nodos pueden venir en forma (3, n) (cada columna es [x,y,z]) o en forma (n, 3).
            Si es (3, n), se transpone a (n, 3).
          - Las caras pueden venir en forma (3, m) (cada columna es la conectividad de un triángulo)
            o en forma (m, 3). Si es (3, m), se transpone a (m, 3).
          - El vector de potenciales corresponde a un instante de tiempo de una señal
            y su longitud debe ser igual al número de nodos de la malla.

        Args:
            nodes: Array de nodos (forma (3, n) o (n, 3)).
            faces: Array de conectividad (forma (3, m) o (m, 3)).
            potentials: Vector de potenciales (longitud igual al número de nodos).
            structure_label: Etiqueta para el título (ejemplo: "Heart", "Torso").
            cmap: Mapa de colores a utilizar (por defecto 'viridis').
        """
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        import matplotlib.pyplot as plt

        # Convertir a arrays de NumPy
        try:
            nodes = np.array(nodes)
            faces = np.array(faces, dtype=int)
            potentials = np.array(potentials, dtype=float)
        except Exception as e:
            logger.error(f"Error converting inputs to NumPy arrays: {e}")
            return

        # Si nodes viene en forma (3, n), transpónalo a (n, 3)
        if nodes.shape[0] == 3 and nodes.shape[1] > 3:
            nodes = nodes.T

        # Verificar que el número de potenciales coincida con el número de nodos
        if potentials.shape[0] != nodes.shape[0]:
            logger.error(f"El número de potenciales ({potentials.shape[0]}) no coincide con el número de nodos ({nodes.shape[0]}).")
            return

        # Si faces viene en forma (3, m), transpónelo a (m, 3)
        if faces.size == 0:
            logger.error("No hay información de caras para construir la malla.")
            return
        if faces.shape[0] == 3 and faces.shape[1] > 3:
            faces = faces.T

        # Ajustar índices si son 1-indexados (MATLAB)
        if np.min(faces) == 1:
            faces = faces - 1

        # Construir la lista de polígonos (aceptamos cualquier cara de forma (3,3))
        polys = []
        for face in faces:
            try:
                poly = nodes[face]
                if poly.shape == (3, 3):  # Aceptamos la cara si tiene 3 nodos con 3 coordenadas cada uno
                    polys.append(poly)
            except Exception as e:
                logger.warning(f"Error processing face {face}: {e}")
        if len(polys) == 0:
            logger.error("No se pudieron extraer polígonos válidos de las caras.")
            return

        logger.info(f"Se extrajeron {len(polys)} polígonos para la malla.")

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Crear la malla (Poly3DCollection) sin relleno, para que se vea el pseudocolor de los nodos.
        mesh = Poly3DCollection(polys, alpha=0.5, edgecolor='k', linewidths=0.5)
        mesh.set_facecolor('none')
        ax.add_collection3d(mesh)

        # Graficar los nodos como puntos, coloreados según el potencial
        sc = ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], c=potentials, cmap=cmap, s=20)
        cbar = plt.colorbar(sc, ax=ax, shrink=0.5)
        cbar.set_label("Potential (mV)")

        # Ajustar límites según los nodos
        margin = 0.1
        ax.set_xlim(np.min(nodes[:, 0]) - margin, np.max(nodes[:, 0]) + margin)
        ax.set_ylim(np.min(nodes[:, 1]) - margin, np.max(nodes[:, 1]) + margin)
        ax.set_zlim(np.min(nodes[:, 2]) - margin, np.max(nodes[:, 2]) + margin)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"{structure_label} Mesh with ECG Potential")
        plt.tight_layout()
        plt.show()