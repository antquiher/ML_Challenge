# ML Challenge — Conducción Autónoma en Duckietown mediante Deep RL

**Autores:** Antonio Quijano Herrera · Carmen Gutiérrez Silva

---

## Resumen del proyecto

El objetivo es entrenar un agente capaz de conducir un Duckiebot en el simulador 3D Duckietown usando únicamente píxeles de la cámara frontal, y que generalice a un mapa oculto con obstáculos (`Duckietown-loop_obstacles-v0`). El trabajo se estructura en tres fases:

### Fase 1 — Q-Learning tabular (FrozenLake-v1 8x8)
Implementación de Q-Learning desde cero (sin librerías de Deep RL) usando la ecuación de Bellman en su forma off-policy. Se compara empíricamente frente a Monte Carlo, demostrando que TD es claramente superior en entornos dispersos y estocásticos: ~70% de éxito de Q-Learning frente a ~3% de Monte Carlo tras 300.000 episodios.

### Fase 2 — Baselines: DQN y PPO
- **DQN**: requiere discretizar el espacio de acción continuo de Duckietown mediante `DiscretizationWrapper`. La tabla de acciones es simétrica, incluye avance recto y **excluye la acción de parar** (ver diagnóstico más abajo).
- **PPO**: trabaja nativamente con acciones continuas; configuración: `lr=3e-4`, `n_steps=512`, `batch_size=64`, `ent_coef=0.01`.

### Fase 3 — Algoritmo avanzado: SAC (Soft Actor-Critic)
SAC se selecciona por ser off-policy (reutiliza experiencias del replay buffer), maximizar entropía (exploración sostenida, política más robusta) y soportar acciones continuas nativas. Configuración: `lr=3e-4`, `buffer_size=100k`, `batch_size=256`, `ent_coef='auto'`.

Todos los agentes comparten el mismo pipeline de percepción (`DuckieWrapper` + `CustomCNN` estilo Nature-DQN: 3 capas conv + capa densa de 256 unidades) y usan `VecFrameStack(n_stack=4)` para percepción de velocidad y dinámica de giro.

---

## Diagnóstico del reward y decisiones de diseño

El hallazgo más importante fue un **bug en la función de recompensa**: el `RewardWrapper` inicial leía `info.get('forward_reward', 0)` e `info.get('lane_position', None)`, pero gym-duckietown publica esa información anidada en `info['Simulator']['lane_position']` y `['robot_speed']`. Como consecuencia:
- Los premios por avanzar y centrarse **nunca se activaban**.
- La penalización por moverse sin avanzar (`reward -= 3.0`) se disparaba en cada paso.
- **Resultado**: moverse costaba −3.0, quedarse quieto solo −0.5 → el agente aprendía a no moverse.

**Decisión de diseño — acción 'parar':** con la señal defectuosa, el agente aprendía que parar era la política óptima (no te sales de la carretera si no te mueves). Por eso se eliminó la acción `[0.0, 0.0]` de `DiscretizationWrapper`, forzando al agente a moverse siempre.

**Corrección identificada:** la solución correcta es acceder a `info['Simulator']`:

```python
class RewardWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        sim_info = info.get('Simulator', {})
        lane_pos = sim_info.get('lane_position', None)
        speed    = sim_info.get('robot_speed', 0.0)
        reward = 0.0
        if lane_pos is not None:
            reward += 1.5 * lane_pos.get('dot_dir', 0)   # alineación con el carril
            reward -= 6.0 * abs(lane_pos.get('dist', 0)) # penalización por desviación
        if speed < 0.05:
            reward -= 1.0    # anti-inmovilidad
        if done:
            reward -= 20.0   # penalización por salida
        return obs, reward, done, truncated, info
```

**Limitación de recursos:** aunque la solución fue identificada con precisión, el re-entrenamiento no pudo ejecutarse:
- **RAM local insuficiente**: SAC con 100k replay buffer y 5 entornos en paralelo supera la RAM disponible.
- **Google Colab**: a pesar de múltiples intentos (incluyendo el fragmento de código del profesor), no fue posible estabilizar el entorno (errores de display virtual, incompatibilidades de versiones, RAM insuficiente en instancia gratuita).

Con más memoria RAM o una instancia Colab Pro, aplicar este reward corregido mejoraría sustancialmente los resultados.

---

## Resultados (pre-fix)

| Algoritmo | Pasos | Tiempo | Recompensa |
|-----------|-------|--------|------------|
| DQN       | 90    | 3.0 s  | −2098      |
| PPO       | 61    | 2.0 s  | −2489      |
| **SAC**   | **106** | **3.5 s** | **−2941** |

SAC fue el más robusto incluso con el reward defectuoso, coherente con su naturaleza off-policy y de máxima entropía. El agente entregado es SAC (`best_duckie_agent.zip`).

---

## Instalación de dependencias

Instala las dependencias exactas antes de ejecutar el notebook:

```bash
pip install -r requirements.txt
```

> Es imprescindible usar las versiones fijadas en `requirements.txt` para garantizar la compatibilidad con el contrato de evaluación del profesor.

---

## Archivos entregados

| Archivo | Descripción |
|---------|-------------|
| `Challenge_RL.ipynb` | Notebook completo con Fases 1, 2 y 3 |
| `requirements.txt` | Dependencias con versiones fijadas |
| `best_duckie_agent.zip` / `sac_duckie_agent.zip` | Agente SAC entrenado (mejor resultado) |
| `dqn_duckie_agent.zip` | Agente DQN entrenado |
| `ppo_duckie_agent.zip` | Agente PPO entrenado |
| `eval_sac.mp4` / `eval_dqn.mp4` / `eval_ppo.mp4` | Vídeos de evaluación |
| `Report.docx` | Memoria técnica completa |
