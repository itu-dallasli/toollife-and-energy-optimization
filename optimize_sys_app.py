import os
import multiprocessing
import sys 
import threading 
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Derin Öğrenme Kütüphaneleri
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# --- 1. DONANIM KULLANIMINI ARTIRAN AYARLAR (KODUN EN BAŞI) ---
try:
    NUM_THREADS = multiprocessing.cpu_count()
except NotImplementedError:
    NUM_THREADS = 4

os.environ["OMP_NUM_THREADS"] = str(NUM_THREADS) 
os.environ["MKL_NUM_THREADS"] = str(NUM_THREADS) 
os.environ["OPENBLAS_NUM_THREADS"] = str(NUM_THREADS)
os.environ["NUMEXPR_NUM_THREADS"] = str(NUM_THREADS)
print(f"Bilimsel kütüphaneler {NUM_THREADS} iş parçacığı kullanmaya ayarlandı.")

try:
    tf.config.threading.set_inter_op_parallelism_threads(NUM_THREADS)
    tf.config.threading.set_intra_op_parallelism_threads(NUM_THREADS)
except Exception as e:
    print(f"TensorFlow threading ayarı başarısız: {e}")

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU ayarı başarısız: {e}")

try:
    import pyi_splash
except ImportError:
    pyi_splash = None 

# --- 2. Veri Seti ve Etiketler ---
data_list = [
    [100, 0.10, 'Dry', 60, 23.29], [100, 0.15, 'Dry', 48, 19.64],
    [150, 0.10, 'Dry', 35, 19.91], [150, 0.15, 'Dry', 32, 17.75],
    [100, 0.10, 'MQL', 72, 24.05], [100, 0.15, 'MQL', 68, 20.51],
    [150, 0.10, 'MQL', 63, 20.13], [150, 0.15, 'MQL', 57, 18.47],
    [100, 0.10, 'Hybrid', 94, 25.56], [100, 0.15, 'Hybrid', 89, 21.17],
    [150, 0.10, 'Hybrid', 75, 20.91], [150, 0.15, 'Hybrid', 66, 19.29],
    [100, 0.10, 'Cryo', 121, 22.63], [100, 0.15, 'Cryo', 162, 18.32],
    [150, 0.10, 'Cryo', 107, 18.51], [150, 0.15, 'Cryo', 101, 17.24],
    [100, 0.10, 'NF-1', 175, 24.42], [100, 0.15, 'NF-1', 157, 22.85],
    [150, 0.10, 'NF-1', 149, 22.46], [150, 0.15, 'NF-1', 103, 20.88],
    [100, 0.10, 'NF-2', 202, 23.26], [100, 0.15, 'NF-2', 200, 21.62],
    [150, 0.10, 'NF-2', 189, 21.44], [150, 0.15, 'NF-2', 170, 19.59],
]

df = pd.DataFrame(data_list, columns=['Vc', 'fn', 'Condition', 'T', 'E'])
ALL_CONDS = df['Condition'].unique().tolist()

COND_LABELS = {
    'Dry': 'Kuru İşleme', 'MQL': 'MQL (Minimum Yağlama)', 
    'Hybrid': 'Hibrit Yağlama', 'Cryo': 'Kriyojenik Soğutma', 
    'NF-1': 'Nanofluid 1', 'NF-2': 'Nanofluid 2'
}

X = df[['Vc', 'fn', 'Condition']]
Y = df[['T', 'E']]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Vc', 'fn']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Condition'])
    ])

X_processed = preprocessor.fit_transform(X)
INPUT_DIM = X_processed.shape[1] 
MLP_MODEL = None 
GUI = {}

# --- 3. MLP Model Tanımlama ve Kaydetme/Yükleme Fonksiyonları ---

def build_mlp_model(input_shape):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_shape,)),
        Dense(32, activation='relu'),
        Dense(2, activation='linear') 
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae']) 
    return model

def train_and_save_new_model(model_path):
    """Modeli eğitir ve kaydeder."""
    global MLP_MODEL
    print("MLP Modeli eğitiliyor (İlk çalıştırma)...")
    
    MLP_MODEL = build_mlp_model(INPUT_DIM)
    MLP_MODEL.fit(X_processed, Y.values, epochs=500, batch_size=4, verbose=0)
    
    try:
        MLP_MODEL.save(model_path)
        print("MLP Modeli Başarıyla Eğitildi ve kaydedildi.")
    except Exception as e:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"KRİTİK HATA: MODEL KAYDI BAŞARISIZ OLDU. Hata Kodu: {e}")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        if GUI.get('root'):
             messagebox.showerror("Kayıt Hatası", f"Model dosyası oluşturulamadı: {e}")
        
def load_or_train_mlp_model():
    """Kayıtlı modeli yüklemeye çalışır, yoksa eğitir ve kaydeder."""
    global MLP_MODEL
    model_path = 'mlp_model.keras'
    
    if os.path.exists(model_path):
        print(f"'{model_path}' dosyası bulundu. Model yükleniyor...")
        try:
            MLP_MODEL = load_model(model_path)
            print("MLP Modeli Başarıyla Yüklendi.")
        except Exception as e:
            print(f"Hata: Model yüklenirken bir sorun oluştu ({e}). Yeniden eğitiliyor...")
            train_and_save_new_model(model_path) 
    else:
        print(f"'{model_path}' dosyası bulunamadı. Model eğitiliyor ve kaydediliyor...")
        train_and_save_new_model(model_path)
        
    if MLP_MODEL is not None:
        try:
            # TF/Keras ilk çağrı gecikmesini giderme
            predict_mlp(125, 0.125, 'Dry') 
        except Exception as e:
            pass
            
    if pyi_splash:
        pyi_splash.close()

# --- 4. Tahmin ve Optimizasyon Fonksiyonları ---

def predict_mlp(Vc, fn, condition):
    if MLP_MODEL is None: return 0, float('inf')
    single_data = pd.DataFrame([[Vc, fn, condition]], columns=['Vc', 'fn', 'Condition'])
    single_data_processed = preprocessor.transform(single_data)
    prediction = MLP_MODEL.predict(single_data_processed, verbose=0)[0]
    return prediction[0], prediction[1]

def objective_func_mlp(params, condition):
    Vc, fn = params
    T, E = predict_mlp(Vc, fn, condition)
    if T <= 0 or E == float('inf'): return 1e9
    return E / T

def optimize(condition):
    Vc_bounds = (80, 170) 
    fn_bounds = (0.08, 0.17)
    bounds_list = [Vc_bounds, fn_bounds]
    x0 = [125, 0.125]
    
    result = minimize(
        objective_func_mlp,
        x0, 
        args=(condition,), 
        method='COBYLA', 
        bounds=bounds_list, 
        options={'maxiter': 5000} 
    )
    
    if result.success:
        opt_Vc, opt_fn = result.x
        opt_T, opt_E = predict_mlp(opt_Vc, opt_fn, condition)
        return {'Vc': opt_Vc, 'fn': opt_fn, 'T': opt_T, 'E': opt_E, 'ratio': opt_E / opt_T}
    return None

# --- 5. GUI & Threading Yapılandırması ---

selected_condition = 'Dry' 

def update_condition_selection(current_condition):
    global selected_condition
    
    if GUI['vars'][current_condition].get() == 1:
        for cond, var in GUI['vars'].items():
            if var.get() == 1 and cond != current_condition:
                var.set(0)
        selected_condition = current_condition
    else:
        if selected_condition == current_condition and sum(v.get() for v in GUI['vars'].values()) == 0:
             selected_condition = None

    if selected_condition:
        button_text = f"'{COND_LABELS[selected_condition]}' için Hesapla (MLP)"
    else:
        button_text = "Lütfen Koşul Seçin"
        
    if GUI.get('calc_button'):
        GUI['calc_button'].config(text=button_text)
        
    if selected_condition and MLP_MODEL is not None and GUI.get('canvas'):
        plot_validation(selected_condition, GUI['ax_t'], GUI['ax_e'])
    elif GUI.get('canvas'):
        GUI['ax_t'].clear(); GUI['ax_t'].set_title("Koşul Seçilmedi", fontsize=10)
        GUI['ax_e'].clear(); GUI['ax_e'].set_title("Koşul Seçilmedi", fontsize=10)
        GUI['canvas'].draw()


def run_calculation():
    selected_conditions = [cond for cond, var in GUI['vars'].items() if var.get() == 1]
    
    if len(selected_conditions) != 1: return False
        
    condition = selected_conditions[0]

    try:
        Vc = float(GUI['Vc_entry'].get())
        fn = float(GUI['fn_entry'].get())
        
        if Vc <= 0 or fn <= 0: return False
        
        T_model, E_model = predict_mlp(Vc, fn, condition)
        opt_result = optimize(condition) 
        
        results = {
            'condition': condition, 'Vc': Vc, 'fn': fn, 
            'T_model': T_model, 'E_model': E_model, 
            'opt_result': opt_result
        }
        
        return results 

    except ValueError:
        messagebox.showerror("Giriş Hatası", "Vc ve fn için geçerli sayısal değerler giriniz.")
        return False
    except Exception as e:
        print(f"Hata: Hesaplama sırasında beklenmeyen bir hata oluştu: {e}")
        return False


def finalize_gui_update(results):
    GUI['calc_button'].config(state=tk.NORMAL) 
    
    if results is False:
        if MLP_MODEL is None:
            messagebox.showerror("Hata", "MLP modeli henüz yüklenmedi/eğitilmedi.")
        update_condition_selection(selected_condition) 
        return
        
    condition = results['condition']
    Vc, fn = results['Vc'], results['fn']
    T_model, E_model = results['T_model'], results['E_model']
    opt_result = results['opt_result']
    
    # --- Sapma Kontrolü ---
    validation_text = ""
    subset_data_row = df[(df['Condition'] == condition) & (abs(df['Vc'] - Vc) < 0.001) & (abs(df['fn'] - fn) < 0.001)]
    
    if not subset_data_row.empty:
        val_T_real = subset_data_row['T'].iloc[0]
        val_E_real = subset_data_row['E'].iloc[0]
        t_perc_error = (abs(val_T_real - T_model) / val_T_real) * 100 if val_T_real else 0
        e_perc_error = (abs(val_E_real - E_model) / val_E_real) * 100 if val_E_real else 0
        validation_text = (
            f"\n--- Deneysel Sapma (MLP) ---\n"
            f"  T (Gerçek/Model): {val_T_real:.2f} s / {T_model:.2f} s\n"
            f"  Sapma Yüzdesi (T): {t_perc_error:.4f} %\n"
            f"  E (Gerçek/Model): {val_E_real:.2f} kJ / {E_model:.2f} kJ\n"
            f"  Sapma Yüzdesi (E): {e_perc_error:.4f} %"
        )
    else:
          validation_text = (
            f"\n--- Deneysel Sapma Kontrolü ---\n"
            f"  Giriş noktası ({Vc:.2f}, {fn:.2f}) deney aralığında, ancak birebir deney noktası değil."
          )

    # --- Sonuç Metni ---
    opt_text = ""
    if opt_result:
        opt_text = (
            f"\n--- Optimizasyon (MLP - Min E/T) ---\n"
            f"  Optimum Vc: {opt_result['Vc']:.2f} m/dk\n"
            f"  Optimum fn: {opt_result['fn']:.3f} mm/dev\n"
            f"  Tahmini T: {opt_result['T']:.2f} s, E: {opt_result['E']:.2f} kJ\n"
            f"  Verimlilik Oranı (E/T): {opt_result['ratio']:.5f}"
        )

    result_text = (
        f"--- Giriş Parametreleri Tahmini (MLP Modeli) ---\n"
        f"Koşul: {COND_LABELS[condition]} (Vc={Vc:.2f}, fn={fn:.2f})\n"
        f"  Tahmini Takım Ömrü (T): {T_model:.2f} s\n"
        f"  Tahmini Enerji (E): {E_model:.2f} kJ\n"
        f"{validation_text}\n"
        f"{opt_text}\n"
        f"\n--- MLP Modeli Notu ---\n"
        f"araç, yapay sinir ağı tabanlı MLP modeli kullanarak talaşlı imalat parametre tahmini ve optimizasyonu yapar. daha fazla veri ile model performansı artabilir."
    )
        
    GUI['res_box'].config(state=tk.NORMAL)
    GUI['res_box'].delete(1.0, tk.END)
    GUI['res_box'].insert(tk.END, result_text)
    GUI['res_box'].config(state=tk.DISABLED)
    
    plot_validation(condition, GUI['ax_t'], GUI['ax_e'])
    update_condition_selection(selected_condition)


def start_calculation_thread(initial_run=False):
    if MLP_MODEL is None and not initial_run:
        messagebox.showinfo("Bilgi", "MLP Modeli henüz yüklenmedi/eğitilmedi. Lütfen kısa bir süre bekleyiniz.")
        return
        
    GUI['calc_button'].config(state=tk.DISABLED, text="HESAPLANIYOR... LÜTFEN BEKLEYİNİZ (Arka Plan İşlemi)")
    
    thread = threading.Thread(target=lambda: thread_wrapper(run_calculation), daemon=True)
    thread.start()

def thread_wrapper(func):
    results = func()
    GUI['root'].after(0, lambda: finalize_gui_update(results))


def plot_validation(condition, ax_t, ax_e):
    subset = df[df['Condition'] == condition]
    fn_010 = subset[subset['fn'] == 0.10]
    fn_015 = subset[subset['fn'] == 0.15]
    Vc_range = np.linspace(80, 170, 100)

    # T Grafiği
    ax_t.clear()
    ax_t.set_title(f'{COND_LABELS[condition]} - T Tahmini (MLP)', fontsize=10)
    ax_t.set_xlabel('Vc (m/dk)', fontsize=8); ax_t.set_ylabel('T (s)', fontsize=8)
    ax_t.tick_params(axis='both', which='major', labelsize=7)
    
    if not fn_010.empty:
        T_model_10 = [predict_mlp(v, 0.10, condition)[0] for v in Vc_range]
        ax_t.scatter(fn_010['Vc'], fn_010['T'], color='blue', label='Deney (fn=0.10)', marker='o')
        ax_t.plot(Vc_range, T_model_10, color='blue', linestyle='-', label='Model (fn=0.10)')

    if not fn_015.empty:
        T_model_15 = [predict_mlp(v, 0.15, condition)[0] for v in Vc_range]
        ax_t.scatter(fn_015['Vc'], fn_015['T'], color='red', label='Deney (fn=0.15)', marker='s')
        ax_t.plot(Vc_range, T_model_15, color='red', linestyle='--', label='Model (fn=0.15)')

    ax_t.legend(fontsize=7, loc='upper right'); ax_t.grid(True, linestyle=':', alpha=0.6)
    
    # E Grafiği
    ax_e.clear()
    ax_e.set_title(f'{COND_LABELS[condition]} - E Tahmini (MLP)', fontsize=10)
    ax_e.set_xlabel('Vc (m/dk)', fontsize=8); ax_e.set_ylabel('E (kJ)', fontsize=8)
    ax_e.tick_params(axis='both', which='major', labelsize=7)

    if not fn_010.empty:
        E_model_10 = [predict_mlp(v, 0.10, condition)[1] for v in Vc_range]
        ax_e.scatter(fn_010['Vc'], fn_010['E'], color='blue', label='Deney (fn=0.10)', marker='o')
        ax_e.plot(Vc_range, E_model_10, color='blue', linestyle='-', label='Model (fn=0.10)')
        
    if not fn_015.empty:
        E_model_15 = [predict_mlp(v, 0.15, condition)[1] for v in Vc_range]
        ax_e.scatter(fn_015['Vc'], fn_015['E'], color='red', label='Deney (fn=0.15)', marker='s')
        ax_e.plot(Vc_range, E_model_15, color='red', linestyle='--', label='Model (fn=0.15)')

    ax_e.legend(fontsize=7, loc='upper right'); ax_e.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    GUI['canvas'].draw()


def create_gui():
    
    root = tk.Tk()
    root.title("Talaşlı İmalat Tahmin ve Optimizasyon Aracı (MLP Modeli)")
    root.geometry("1100x900")
    GUI['root'] = root 

    main_frame = ttk.Frame(root, padding="10"); main_frame.pack(fill=tk.BOTH, expand=True)
    paned_window = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL); paned_window.pack(fill=tk.BOTH, expand=True)
    left_frame = ttk.Frame(paned_window, padding="5"); paned_window.add(left_frame, weight=0)
    input_frame = ttk.LabelFrame(left_frame, text="Parametre Girişi", padding="10"); input_frame.pack(fill='x', pady=10)

    # Vc Girişi
    ttk.Label(input_frame, text="Vc (m/dk):").grid(row=0, column=0, padx=5, pady=5, sticky='w')
    GUI['Vc_entry'] = ttk.Entry(input_frame, width=10); GUI['Vc_entry'].insert(0, "125.00"); GUI['Vc_entry'].grid(row=0, column=1, padx=5, sticky='ew')
    
    # fn Girişi
    ttk.Label(input_frame, text="fn (mm/dev):").grid(row=1, column=0, padx=5, pady=5, sticky='w')
    GUI['fn_entry'] = ttk.Entry(input_frame, width=10); GUI['fn_entry'].insert(0, "0.125"); GUI['fn_entry'].grid(row=1, column=1, padx=5, sticky='ew')
    
    # Koşul Seçim Çerçevesi
    condition_frame = ttk.LabelFrame(left_frame, text="İşleme Koşulu", padding="10"); condition_frame.pack(fill='x', pady=10)
    GUI['vars'] = {}
    for i, condition in enumerate(ALL_CONDS):
        GUI['vars'][condition] = tk.IntVar()
        if condition == 'Dry': GUI['vars'][condition].set(1)
        chk = ttk.Checkbutton(condition_frame, text=COND_LABELS[condition], variable=GUI['vars'][condition], command=lambda c=condition: update_condition_selection(c))
        chk.grid(row=i // 2, column=i % 2, padx=10, pady=5, sticky='w')
        
    GUI['calc_button'] = ttk.Button(
        left_frame, text=f"MLP Modeli Yükleniyor...", command=start_calculation_thread, state=tk.DISABLED
    )
    GUI['calc_button'].pack(pady=15, fill='x')

    ttk.Label(left_frame, text="Sonuçlar:").pack(fill='x')
    GUI['res_box'] = tk.Text(left_frame, height=18, width=70, state=tk.DISABLED, bg='lightgray')
    GUI['res_box'].pack(fill='both', expand=True)

    right_frame = ttk.Frame(paned_window, padding="5"); paned_window.add(right_frame, weight=1)
    
    ttk.Label(right_frame, text="MLP Model Doğrulama Grafikleri (Deney vs. Tahmin)").pack(pady=5)
    
    GUI['fig'], (GUI['ax_t'], GUI['ax_e']) = plt.subplots(2, 1, figsize=(6, 8))
    GUI['canvas'] = FigureCanvasTkAgg(GUI['fig'], master=right_frame)
    canvas_widget = GUI['canvas'].get_tk_widget()
    canvas_widget.pack(fill=tk.BOTH, expand=True)
    
    # Model Yükleme/Eğitme İşlemi (Arka Plan)
    threading.Thread(target=load_or_train_mlp_model, daemon=True).start()
    
    def check_model_ready():
        if MLP_MODEL is not None:
            GUI['calc_button'].config(state=tk.NORMAL, text=f"'{COND_LABELS[selected_condition]}' için Hesapla (MLP)")
            update_condition_selection(selected_condition)
        else:
            root.after(100, check_model_ready)
            
    root.after(100, check_model_ready)
    root.mainloop()

if __name__ == "__main__":
    create_gui()