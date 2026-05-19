# ObstacleFeatures.msg
# Posicion en imagen del obstaculo detectado por color.
# Publicado por target_detector.py, consumido por vs_controller.py.
#
# El detector reporta solo el blob mas grande del color "obstaculo"
# (sea cual sea: magenta, naranja, azul, segun config).  Si hay varios
# blobs el resto se ignora.

std_msgs/Header header

# True si se detecto un blob del color del obstaculo en este frame.
bool detected

# Centroide horizontal del blob, en pixeles.  Indefinido si detected==false.
float32 u_c
float32 v_c

# Raiz del area en pixeles.  Cero si no detectado.
# Sirve como proxy de distancia al obstaculo (mas grande -> mas cerca).
float32 sqrt_area

# Area cruda en pixeles, para diagnostico y CSV.
float32 area_px

float32 solidity
