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
    # Application pour réduire les étoiles de manière interactive
    
    def __init__(self, root):
        self.root = root
        self.root.title("Phase 3: Réduction Interactive des Étoiles")
        self.root.geometry("1400x900")
        
        # Variables pour les images
        self.data_gray = None
        self.data_original_color = None  
        self.image_original = None
        self.image_final = None
        self.mean = 0
        self.median = 0
        self.std = 0
        self.sources = None
        
        # Paramètres par défaut
        self.threshold_multiplier = 3.0
        self.star_radius = 3
        self.gaussian_size = 5
        self.gaussian_sigma = 1
        self.erosion_kernel = 3
        self.erosion_iterations = 1
        
        # Pour le clignotement
        self.blink_active = False
        self.blink_timer = None
        self.show_original = True
        
        self.create_interface()
    
    def create_interface(self):
        # Frame gauche: contrôles
        left_frame = ttk.Frame(self.root, width=300)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        # Titre en haut
        title = ttk.Label(left_frame, text="Réduction Interactive", font=('Arial', 14, 'bold'))
        title.pack(pady=10)
        
        # Bouton pour charger le fichier FITS
        ttk.Button(left_frame, text="Charger FITS", command=self.load_fits).pack(pady=5, fill=tk.X)
        
        ttk.Separator(left_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # Paramètres
        params_frame = ttk.LabelFrame(left_frame, text="Paramètres", padding=10)
        params_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.sliders = {}
        
        # Seuil de détection
        self.add_slider(params_frame, "Seuil détection (×σ)", 'threshold_multiplier', 1.0, 5.0, 3.0)
        
        # Rayon étoiles
        self.add_slider(params_frame, "Rayon étoiles (px)", 'star_radius', 2, 12, 3)
        
        # Flou gaussien taille
        self.add_slider(params_frame, "Flou gaussien (taille)", 'gaussian_size', 3, 15, 5)
        
        # Flou gaussien sigma
        self.add_slider(params_frame, "Flou gaussien (σ)", 'gaussian_sigma', 1, 10, 1)
        
        # Kernel érosion
        self.add_slider(params_frame, "Kernel érosion", 'erosion_kernel', 3, 9, 3)
        
        # Itérations érosion
        self.add_slider(params_frame, "Itérations érosion", 'erosion_iterations', 1, 3, 1)
        
        # Bouton appliquer
        ttk.Button(left_frame, text="Appliquer", command=self.apply_processing).pack(pady=10, fill=tk.X)
        
        ttk.Separator(left_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # Section Comparateur
        comp_frame = ttk.LabelFrame(left_frame, text="Comparateur Avant/Après", padding=10)
        comp_frame.pack(fill=tk.X, pady=5)
        
        # Bouton clignotement
        self.blink_button = ttk.Button(comp_frame, text="▶ Clignotement", command=self.toggle_blink)
        self.blink_button.pack(pady=5, fill=tk.X)
        
        # Slider de fondu
        ttk.Label(comp_frame, text="Fondu (0%=original, 100%=résultat)").pack(pady=(10,0))
        self.blend_var = tk.DoubleVar(value=100)
        self.blend_slider = ttk.Scale(comp_frame, from_=0, to=100, orient=tk.HORIZONTAL, 
                                      variable=self.blend_var, command=self.update_blend)
        self.blend_slider.pack(fill=tk.X, pady=5)
        
        # Checkbox différence
        self.show_diff_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(comp_frame, text="Afficher différence", 
                       variable=self.show_diff_var, command=self.update_display).pack(pady=5)
        
        ttk.Separator(left_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # Info
        self.info_label = ttk.Label(left_frame, text="Chargez une image FITS", wraplength=280)
        self.info_label.pack(pady=10)
        
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
        # Crée un slider avec son label
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
        # Charger un fichier FITS
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
            
            # Gestion des images couleur vs monochrome
            if data.ndim == 3:
                if data.shape[0] == 3:
                    data = np.transpose(data, (1, 2, 0))
                self.data_gray = np.mean(data, axis=2)
                # Garder la version couleur
                data_min, data_max = data.min(), data.max()
                self.data_original_color = (data - data_min) / (data_max - data_min)
            else:
                self.data_gray = data.copy()
                self.data_original_color = None
            
            # Normaliser en float64
            gmin = self.data_gray.min()
            gmax = self.data_gray.max()
            data_normalized = (self.data_gray - gmin) / (gmax - gmin)
            self.image_original = data_normalized.astype(np.float64)
            
            # Statistiques du fond
            self.mean, self.median, self.std = sigma_clipped_stats(self.data_gray, sigma=3.0)
            
            hdul.close()
            
            # Détection initiale
            self.detect_stars()
            
            # Afficher
            filename = os.path.basename(filepath)
            nb_etoiles = len(self.sources) if self.sources else 0
            self.info_label.config(
                text=f"✓ {filename}\n"
                     f"Dimensions: {self.image_original.shape}\n"
                     f"Étoiles: {nb_etoiles}\n"
                     f"Fond: μ={self.mean:.3f}, σ={self.std:.3f}"
            )
            
            self.apply_processing()
            
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de charger:\n{str(e)}")
    
    def detect_stars(self):
        # Détection des étoiles avec DAOStarFinder
        threshold = self.threshold_multiplier * self.std
        daofind = DAOStarFinder(fwhm=3.0, threshold=threshold)
        self.sources = daofind(self.data_gray - self.median)
        
        if self.sources is None:
            # Si ça marche pas, on utilise un seuillage simple
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
        # Applique le traitement avec les paramètres
        if self.image_original is None:
            messagebox.showwarning("Attention", "Chargez d'abord une image")
            return
        
        # Récup les valeurs des sliders
        self.threshold_multiplier = self.sliders['threshold_multiplier'].get()
        self.star_radius = int(self.sliders['star_radius'].get())
        self.gaussian_size = int(self.sliders['gaussian_size'].get())
        self.gaussian_sigma = int(self.sliders['gaussian_sigma'].get())
        self.erosion_kernel = int(self.sliders['erosion_kernel'].get())
        self.erosion_iterations = int(self.sliders['erosion_iterations'].get())
        
        # Redetecter les étoiles
        self.detect_stars()
        
        # Création du masque
        mask = np.zeros_like(self.data_gray, dtype=np.float64)
        
        if self.sources is not None:
            for source in self.sources:
                x = int(source['xcentroid'])
                y = int(source['ycentroid'])
                cv.circle(mask, (x, y), radius=self.star_radius, color=1.0, thickness=-1)
        
        # Appliquer le flou
        gsize = self.gaussian_size
        if gsize % 2 == 0:
            gsize += 1
        mask_blur = cv.GaussianBlur(mask, (gsize, gsize), self.gaussian_sigma)
        
        if mask_blur.max() > 0:
            mask_blur = mask_blur / mask_blur.max()
        
        # Érosion
        kernel = np.ones((self.erosion_kernel, self.erosion_kernel), np.float64)
        image_eroded = cv.erode(self.image_original, kernel, iterations=self.erosion_iterations)
        
        # Combiner: I_final = M × I_erode + (1-M) × I_original
        self.image_final = (mask_blur * image_eroded) + ((1 - mask_blur) * self.image_original)
        
        # Si image couleur, appliquer à chaque canal
        if self.data_original_color is not None:
            self.image_final_display = np.zeros_like(self.data_original_color)
            for i in range(3):
                channel_orig = self.data_original_color[:, :, i]
                channel_erod = cv.erode(channel_orig, kernel, iterations=self.erosion_iterations)
                self.image_final_display[:, :, i] = (mask_blur * channel_erod) + ((1 - mask_blur) * channel_orig)
        else:
            self.image_final_display = self.image_final
        
        # Afficher
        self.update_display()
    
    
    def toggle_blink(self):
        """Active/désactive le clignotement"""
        self.blink_active = not self.blink_active
        
        if self.blink_active:
            self.blink_button.config(text="⏸ Arrêter")
            self.blend_slider.config(state='disabled')
            self.do_blink()
        else:
            self.blink_button.config(text="▶ Clignotement")
            self.blend_slider.config(state='normal')
            if self.blink_timer:
                self.root.after_cancel(self.blink_timer)
            # Recréer les axes normaux
            self.fig.clear()
            self.ax_original, self.ax_final = self.fig.subplots(1, 2)
            self.update_display()
    
    def do_blink(self):
        """Effectue le clignotement"""
        if not self.blink_active or self.image_final is None:
            return
        
        self.show_original = not self.show_original
        
        # Effacer la figure complètement et créer un seul axe centré
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        
        # Sélectionner l'image à afficher
        if self.show_original:
            if self.data_original_color is not None:
                img = self.data_original_color
            else:
                img = self.image_original
            title = 'IMAGE ORIGINALE'
            color = 'blue'
        else:
            if self.data_original_color is not None:
                img = self.image_final_display
            else:
                img = self.image_final
            num_stars = len(self.sources) if self.sources else 0
            title = f'IMAGE TRAITÉE ({num_stars} étoiles réduites)'
            color = 'green'
        
        # Afficher l'image centrée
        if self.data_original_color is not None:
            ax.imshow(img)
        else:
            ax.imshow(img, cmap='gray')
        
        ax.set_title(title, color=color, weight='bold', fontsize=16)
        ax.axis('off')
        
        self.fig.tight_layout()
        self.canvas.draw()
        
        # Rappeler après 500ms
        self.blink_timer = self.root.after(500, self.do_blink)
    
    def update_blend(self, value=None):
        """Met à jour le fondu"""
        if self.image_final is None:
            return
        
        self.update_display()
    
    def update_display(self):
        """Met à jour l'affichage"""
        self.ax_original.clear()
        self.ax_final.clear()
        
        # Mode différence
        if self.show_diff_var.get() and self.image_final is not None:
            if self.data_original_color is not None:
                diff = np.abs(self.data_original_color - self.image_final_display)
            else:
                diff = np.abs(self.image_original - self.image_final)
            
            self.ax_original.imshow(self.image_original if self.data_original_color is None else self.data_original_color)
            self.ax_original.set_title('Original')
            
            self.ax_final.imshow(diff, cmap='hot')
            self.ax_final.set_title('Différence (ce qui a changé)')
        
        # Mode fondu
        elif self.image_final is not None:
            blend_ratio = self.blend_var.get() / 100.0
            
            if self.data_original_color is not None:
                blended = blend_ratio * self.image_final_display + (1 - blend_ratio) * self.data_original_color
                self.ax_original.imshow(self.data_original_color)
                self.ax_final.imshow(blended)
            else:
                blended = blend_ratio * self.image_final + (1 - blend_ratio) * self.image_original
                self.ax_original.imshow(self.image_original, cmap='gray')
                self.ax_final.imshow(blended, cmap='gray')
            
            self.ax_original.set_title('Image Originale')
            num_stars = len(self.sources) if self.sources else 0
            self.ax_final.set_title(f'Fondu {int(blend_ratio*100)}% ({num_stars} étoiles)')
        
        else:
            # Pas encore d'image finale
            if self.data_original_color is not None:
                self.ax_original.imshow(self.data_original_color)
            elif self.image_original is not None:
                self.ax_original.imshow(self.image_original, cmap='gray')
            self.ax_original.set_title('Image Originale')
        
        self.ax_original.axis('off')
        self.ax_final.axis('off')
        
        self.fig.tight_layout()
        self.canvas.draw()



def main():
    root = tk.Tk()
    
    # Style
    style = ttk.Style()
    style.theme_use('clam')
    
    app = StarReductionApp(root)
    
    root.mainloop()


if __name__ == "__main__":
    main()
