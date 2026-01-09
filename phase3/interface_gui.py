import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from astropy.io import fits
from photutils.detection import DAOStarFinder
from astropy.stats import sigma_clipped_stats
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import cv2 as cv
import numpy as np
import os


class StarReductionApp:
    """Application de réduction interactive des étoiles"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Phase 3: Réduction Interactive des Étoiles")
        self.root.geometry("1400x900")
        
        # Données
        self.data_gray = None
        self.data_original_color = None  # Pour garder la version couleur
        self.image_original = None
        self.image_final = None
        self.mean = 0
        self.median = 0
        self.std = 0
        self.sources = None
        
        # Paramètres par défaut (simplifiés et plus doux)
        self.threshold_multiplier = 3.0
        self.star_radius = 3
        self.gaussian_size = 5
        self.gaussian_sigma = 1
        self.erosion_kernel = 3
        self.erosion_iterations = 1
        
        self.create_interface()
    
    def create_interface(self):
        # Frame gauche: contrôles
        left_frame = ttk.Frame(self.root, width=300)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        # Titre
        title = ttk.Label(left_frame, text="Réduction Interactive", font=('Arial', 14, 'bold'))
        title.pack(pady=10)
        
        # Bouton charger
        ttk.Button(left_frame, text="Charger FITS", command=self.load_fits).pack(pady=5, fill=tk.X)
        
        ttk.Separator(left_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # Paramètres
        params_frame = ttk.LabelFrame(left_frame, text="Paramètres", padding=10)
        params_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.sliders = {}
        
        # Rayon étoiles (plus petit par défaut)
        self.add_slider(params_frame, "Rayon étoiles (px)", 'star_radius', 2, 12, 3)
        
        # Flou du masque (plus doux par défaut)
        self.add_slider(params_frame, "Flou du masque", 'gaussian_size', 3, 15, 5)
        
        # Force de l'érosion (gardé à 1)
        self.add_slider(params_frame, "Force de l'érosion", 'erosion_iterations', 1, 3, 1)
        
        # Bouton appliquer
        ttk.Button(left_frame, text="Appliquer", command=self.apply_processing).pack(pady=10, fill=tk.X)
        
        # Info
        self.info_label = ttk.Label(left_frame, text="Chargez une image FITS", wraplength=280)
        self.info_label.pack(pady=10)
        
        # Bouton sauvegarder
        ttk.Button(left_frame, text="Sauvegarder", command=self.save_result).pack(pady=5, fill=tk.X)
        
        # Frame droite: visualisation
        right_frame = ttk.Frame(self.root)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Figure matplotlib avec 2 images côte à côte
        self.fig = Figure(figsize=(12, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Deux axes: original et résultat
        self.ax_original, self.ax_final = self.fig.subplots(1, 2)
        self.fig.suptitle('Comparaison Original / Résultat', fontsize=14)
        
        self.ax_original.axis('off')
        self.ax_final.axis('off')
    
    def add_slider(self, parent, label, param_name, from_, to, default):
        """Ajoute un slider"""
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(frame, text=label, width=20).pack(side=tk.LEFT)
        
        var = tk.DoubleVar(value=default)
        slider = ttk.Scale(frame, from_=from_, to=to, orient=tk.HORIZONTAL, variable=var)
        slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        value_label = ttk.Label(frame, text=f"{default:.1f}", width=5)
        value_label.pack(side=tk.LEFT)
        
        def update_label(*args):
            value_label.config(text=f"{var.get():.1f}")
        
        var.trace('w', update_label)
        self.sliders[param_name] = var
    
    def load_fits(self):
        """Charge un fichier FITS"""
        filepath = filedialog.askopenfilename(
            title="Sélectionner un fichier FITS",
            filetypes=[("FITS files", "*.fits"), ("All files", "*.*")],
            initialdir="../examples"
        )
        
        if not filepath:
            return
        
        try:
            hdul = fits.open(filepath)
            data = hdul[0].data
            
            # Traiter les images couleur
            if data.ndim == 3:
                if data.shape[0] == 3:
                    data = np.transpose(data, (1, 2, 0))
                self.data_gray = np.mean(data, axis=2)
                # Garder la version couleur
                self.data_original_color = (data - data.min()) / (data.max() - data.min())
            else:
                self.data_gray = data.copy()
                self.data_original_color = None
            
            # Normaliser en float64
            data_normalized = (self.data_gray - self.data_gray.min()) / (self.data_gray.max() - self.data_gray.min())
            self.image_original = data_normalized.astype(np.float64)
            
            # Statistiques du fond
            self.mean, self.median, self.std = sigma_clipped_stats(self.data_gray, sigma=3.0)
            
            hdul.close()
            
            # Détection initiale
            self.detect_stars()
            
            # Afficher
            filename = os.path.basename(filepath)
            self.info_label.config(
                text=f"✓ {filename}\n"
                     f"Dimensions: {self.image_original.shape}\n"
                     f"Étoiles: {len(self.sources) if self.sources else 0}\n"
                     f"Fond: μ={self.mean:.3f}, σ={self.std:.3f}"
            )
            
            self.apply_processing()
            
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de charger:\n{str(e)}")
    
    def detect_stars(self):
        """Détecte les étoiles"""
        threshold = self.threshold_multiplier * self.std
        daofind = DAOStarFinder(fwhm=3.0, threshold=threshold)
        self.sources = daofind(self.data_gray - self.median)
        
        if self.sources is None:
            # Fallback
            threshold_value = np.percentile(self.data_gray, 99.5)
            mask_temp = (self.data_gray > threshold_value).astype(np.uint8)
            num_labels, labels = cv.connectedComponents(mask_temp)
            self.sources = []
            
            for label in range(1, num_labels):
                coords = np.where(labels == label)
                if len(coords[0]) > 5:
                    y_center = int(np.mean(coords[0]))
                    x_center = int(np.mean(coords[1]))
                    self.sources.append({'xcentroid': x_center, 'ycentroid': y_center})
    
    def apply_processing(self):
        """Applique le traitement avec les paramètres actuels"""
        if self.image_original is None:
            messagebox.showwarning("Attention", "Chargez d'abord une image")
            return
        
        # Récupérer les paramètres essentiels
        self.star_radius = int(self.sliders['star_radius'].get())
        self.gaussian_size = int(self.sliders['gaussian_size'].get())
        self.erosion_iterations = int(self.sliders['erosion_iterations'].get())
        
        # Redétecter les étoiles
        self.detect_stars()
        
        # Créer le masque
        mask = np.zeros_like(self.data_gray, dtype=np.float64)
        
        if self.sources is not None:
            for source in self.sources:
                x, y = int(source['xcentroid']), int(source['ycentroid'])
                cv.circle(mask, (x, y), radius=self.star_radius, color=1.0, thickness=-1)
        
        # Flou gaussien
        gsize = self.gaussian_size
        if gsize % 2 == 0:
            gsize += 1
        mask_blurred = cv.GaussianBlur(mask, (gsize, gsize), 1)  # Sigma réduit à 1
        
        if mask_blurred.max() > 0:
            mask_blurred = mask_blurred / mask_blurred.max()
        
        # Érosion
        kernel = np.ones((self.erosion_kernel, self.erosion_kernel), np.float64)
        image_eroded = cv.erode(self.image_original, kernel, iterations=self.erosion_iterations)
        
        # Combiner: I_final = M × I_erode + (1-M) × I_original
        self.image_final = (mask_blurred * image_eroded) + ((1 - mask_blurred) * self.image_original)
        
        # Si image couleur, appliquer le masque à chaque canal
        if self.data_original_color is not None:
            self.image_final_display = np.zeros_like(self.data_original_color)
            for i in range(3):
                channel_original = self.data_original_color[:, :, i]
                channel_eroded = cv.erode(channel_original, kernel, iterations=self.erosion_iterations)
                self.image_final_display[:, :, i] = (mask_blurred * channel_eroded) + ((1 - mask_blurred) * channel_original)
        else:
            self.image_final_display = self.image_final
        
        # Afficher
        self.update_display()
    
    def update_display(self):
        """Met à jour l'affichage"""
        self.ax_original.clear()
        self.ax_final.clear()
        
        # Afficher en couleur si disponible
        if self.data_original_color is not None:
            self.ax_original.imshow(self.data_original_color)
            self.ax_final.imshow(self.image_final_display)
        else:
            self.ax_original.imshow(self.image_original, cmap='gray')
            self.ax_final.imshow(self.image_final, cmap='gray')
        
        self.ax_original.set_title('Image Originale')
        self.ax_original.axis('off')
        
        num_stars = len(self.sources) if self.sources else 0
        self.ax_final.set_title(f'Résultat Final ({num_stars} étoiles réduites)')
        self.ax_final.axis('off')
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def save_result(self):
        """Sauvegarde le résultat"""
        if self.image_final is None:
            messagebox.showwarning("Attention", "Aucun résultat à sauvegarder")
            return
        
        filepath = filedialog.asksaveasfilename(
            title="Sauvegarder le résultat",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
            initialdir="./results"
        )
        
        if filepath:
            # Sauvegarder en couleur si disponible
            if hasattr(self, 'image_final_display') and self.data_original_color is not None:
                plt.imsave(filepath, self.image_final_display)
            else:
                plt.imsave(filepath, self.image_final, cmap='gray')
            messagebox.showinfo("Succès", f"Image sauvegardée:\n{filepath}")



def main():
    root = tk.Tk()
    
    # Style
    style = ttk.Style()
    style.theme_use('clam')
    
    app = StarReductionApp(root)
    
    root.mainloop()


if __name__ == "__main__":
    main()
