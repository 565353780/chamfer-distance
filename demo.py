from chamfer_distance.Demo.chamfer_distances import demo as demo_check_algo
from chamfer_distance.Demo.speed_manager import (
    demo_fps as demo_test_fps,
    demo_balance as demo_test_balance,
    demo_speed as demo_test_speed,
)

if __name__ == '__main__':
    demo_check_algo()
    demo_test_fps()
    demo_test_balance()
    demo_test_speed()
