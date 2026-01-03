import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm

# ============= МЕТОД: TEMPORAL COHERENCE =============

def find_shift_between_frames(frame1, frame2, search_range=None):
    """
    Знайти зсув між двома кадрами що максимізує схожість
    
    Args:
        frame1: перший кадр (N,)
        frame2: другий кадр (N,)
        search_range: діапазон пошуку (±range), якщо None — пошук по всьому N
    
    Returns:
        best_shift: оптимальний зсув
        best_similarity: найкраща схожість
    """
    N = len(frame2)
    
    if search_range is None:
        search_range = N // 2  # Пошук по половині
    
    best_shift = 0
    best_similarity = -jnp.inf
    
    # Перебираємо можливі зсуви
    for shift in range(-search_range, search_range + 1):
        frame2_shifted = jnp.roll(frame2, shift)
        
        # Метрика схожості: негативна MSE (або можна correlation)
        #similarity = -jnp.mean((frame1 - frame2_shifted)**2)
        similarity = -jnp.mean((frame2_shifted - frame1) / 0.1)
        
        # Або normalized cross-correlation (краще!)
        f1_centered = frame1 - jnp.mean(frame1)
        f2_centered = frame2_shifted - jnp.mean(frame2_shifted)
        similarity = jnp.sum(f1_centered * f2_centered) / (
            jnp.sqrt(jnp.sum(f1_centered**2) * jnp.sum(f2_centered**2)) + 1e-10
        )
        
        if similarity > best_similarity:
            best_similarity = similarity
            best_shift = shift
    
    return best_shift, best_similarity


def restore_by_temporal_coherence(trj_pert, search_range=50, use_forward_backward=True):
    """
    Відновити зсуви максимізуючи temporal coherence
    
    Ідея: сусідні кадри мають бути максимально схожі
    
    Args:
        trj_pert: пертурбована траєкторія (T, N)
        search_range: максимальний зсув для пошуку
        use_forward_backward: використати forward+backward pass для точності
    
    Returns:
        trj_restored: відновлена траєкторія
        shifts_found: знайдені зсуви (відносні)
        shifts_absolute: абсолютні зсуви (накопичені)
    """
    T, N = trj_pert.shape
    
    print("\n" + "="*60)
    print("🔍 TEMPORAL COHERENCE RESTORATION")
    print("="*60)
    print(f"Search range: ±{search_range} pixels")
    print(f"Forward-backward: {use_forward_backward}")
    
    # ===== FORWARD PASS =====
    print("\n➡️  Forward pass...")
    
    shifts_relative_fwd = [0]  # Перший кадр — референс
    trj_aligned_fwd = [trj_pert[0]]
    
    for t in tqdm(range(1, T), desc="Forward"):
        # Порівнюємо з попереднім ВІДНОВЛЕНИМ кадром
        reference = trj_aligned_fwd[-1]
        current = trj_pert[t]
        
        # Знайти зсув
        shift, similarity = find_shift_between_frames(
            reference, 
            current, 
            search_range=search_range
        )
        
        # Застосувати зсув
        aligned = jnp.roll(current, shift)
        
        trj_aligned_fwd.append(aligned)
        shifts_relative_fwd.append(shift)
    
    trj_aligned_fwd = jnp.stack(trj_aligned_fwd)
    shifts_relative_fwd = jnp.array(shifts_relative_fwd)
    
    if not use_forward_backward:
        # Тільки forward pass
        shifts_absolute = jnp.cumsum(shifts_relative_fwd) % N
        return trj_aligned_fwd, shifts_relative_fwd, shifts_absolute
    
    # ===== BACKWARD PASS =====
    print("\n⬅️  Backward pass...")
    
    shifts_relative_bwd = [0]  # Останній кадр — референс
    trj_aligned_bwd = [trj_pert[-1]]
    
    for t in tqdm(range(T-2, -1, -1), desc="Backward"):
        # Порівнюємо з наступним ВІДНОВЛЕНИМ кадром
        reference = trj_aligned_bwd[-1]
        current = trj_pert[t]
        
        shift, similarity = find_shift_between_frames(
            reference,
            current,
            search_range=search_range
        )
        
        aligned = jnp.roll(current, shift)
        
        trj_aligned_bwd.append(aligned)
        shifts_relative_bwd.append(shift)
    
    trj_aligned_bwd = jnp.stack(trj_aligned_bwd[::-1])  # Reverse
    shifts_relative_bwd = jnp.array(shifts_relative_bwd[::-1])
    
    # ===== MERGE FORWARD & BACKWARD =====
    print("\n🔄 Merging forward and backward passes...")
    
    # Weighted average (більше ваги на forward на початку, backward на кінці)
    weights_fwd = jnp.linspace(1.0, 0.0, T)
    weights_bwd = jnp.linspace(0.0, 1.0, T)
    
    trj_merged = (
        trj_aligned_fwd * weights_fwd[:, None] + 
        trj_aligned_bwd * weights_bwd[:, None]
    )
    
    # Для зсувів беремо forward (або можна середнє)
    shifts_absolute = jnp.cumsum(shifts_relative_fwd) % N
    
    print("\n✅ Temporal coherence restoration complete!")
    
    return trj_merged, shifts_relative_fwd, shifts_absolute


def compute_temporal_smoothness(trj):
    """Обчислити міру temporal smoothness (менше = краще)"""
    T, N = trj.shape
    smoothness = 0
    for t in range(T-1):
        smoothness += jnp.mean((trj[t+1] - trj[t])**2)
    return smoothness / (T-1)


# ============= ГОЛОВНИЙ КОД =============

filename = '1_noisy_0.1'

# Завантажити дані
with h5py.File(f'data/{filename}.h5', 'r') as f:
    trj_pert1 = jnp.array(f['u'][:])

trj_restored1, shifts_rel1, shifts_abs1 = restore_by_temporal_coherence(
    trj_pert1,
    search_range=len(trj_pert1[0, :])//2,
    use_forward_backward=True
)

# save denoised data
with h5py.File(f"data/{filename}_denoised_color.h5", "w") as f:
	f.create_dataset("u", data=trj_restored1)

# plot original data and denoised data
plt.figure(figsize=(20, 15))
plt.subplot(3, 1, 1)
plt.imshow(trj_pert1.T, aspect='auto', cmap='RdBu')
plt.title(f"{filename} Data")

plt.subplot(3, 1, 2)
plt.imshow(trj_restored1.T, aspect='auto', cmap='RdBu')
plt.title(f"Denoised {filename} Data")
plt.tight_layout()

plt.subplot(3, 1, 3)
plt.imshow((abs(trj_pert1-trj_restored1)).T, aspect='auto', cmap='Reds')
plt.title(f"Difference")
plt.tight_layout()
plt.show()