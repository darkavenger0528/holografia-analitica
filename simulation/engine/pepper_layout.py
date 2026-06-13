"""
simulation/engine/pepper_layout.py

Composición del layout para proyección Pepper's Ghost con vidrio
horizontal a 45°. La pantalla está boca abajo, el vidrio encima.

Para este setup solo se necesita UNA vista centrada sobre fondo negro.
El vidrio a 45° refleja la imagen hacia el observador que mira de frente.

La imagen proyectada debe estar:
- Fondo completamente negro (el negro no refleja → transparente)
- Contenido brillante centrado (colores saturados, alto contraste)
- Orientación: normal si la pantalla está boca arriba mirando al vidrio
"""

import pygame

# Proporción del área activa respecto al canvas total
CONTENT_RATIO = 0.75   # 75% del ancho/alto para el modelo


def create_pepper_surface(screen_w: int, screen_h: int):
    """
    Crea la superficie base negra y el rectángulo donde
    debe dibujarse el modelo.

    Retorna
    -------
    base    : Surface negra del tamaño de la pantalla
    content : Rect centrado donde va el modelo
    """
    base = pygame.Surface((screen_w, screen_h))
    base.fill((0, 0, 0))

    cw = int(screen_w * CONTENT_RATIO)
    ch = int(screen_h * CONTENT_RATIO)
    cx = (screen_w - cw) // 2
    cy = (screen_h - ch) // 2

    content_rect = pygame.Rect(cx, cy, cw, ch)
    return base, content_rect


def blit_to_pepper(screen, model_surface, screen_w: int, screen_h: int):
    """
    Pega el modelo renderizado en el centro de la pantalla
    sobre fondo negro. Llama a esto en cada frame.
    """
    screen.fill((0, 0, 0))

    cw = int(screen_w * CONTENT_RATIO)
    ch = int(screen_h * CONTENT_RATIO)
    scaled = pygame.transform.smoothscale(model_surface, (cw, ch))

    cx = (screen_w - cw) // 2
    cy = (screen_h - ch) // 2
    screen.blit(scaled, (cx, cy))
