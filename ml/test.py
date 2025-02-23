import torch
from ConnectedRaods3 import MumbaiEvacuationRouter

def test_evacuation_routes():
    # Initialize router with saved model
    router = MumbaiEvacuationRouter(load_model=True, use_gpu=True)
    
    # Test cases covering different scenarios
    test_locations = [
        # High-risk areas
        # (19.0596, 72.8295, "Dharavi"),  # Known flood-prone area
        # (19.1890, 72.9754, "Thane"),    # Red zone
        
        # Medium-risk areas
        # (19.0330, 72.8457, "Worli"),    # Yellow zone
        # (19.1136, 72.8900, "Andheri"),  # Commercial area
        
        # Safe areas (testing evacuation from safe to safer zones)
        # (19.1741, 72.9519, "Powai"),    # Generally safer area
        
        # Edge cases
        (19.2335, 73.1305, "Kalyan"),   # Far from city center
        # (19.0282, 72.8565, "Malabar Hill")  # Coastal area
    ]
    
    print("\nTesting Evacuation Routes:")
    print("=" * 50)
    
    for location in test_locations:
        print(f"\nTesting evacuation from {location[2]}")
        print("-" * 30)
        
        route, stats = router.find_best_evacuation_route(location)
        
        if route and stats:
            print(f"Route found!")
            print(f"Total distance: {stats['total_distance']:.2f} meters")
            print("Zone distribution:")
            for zone, percentage in stats['zone_percentages'].items():
                print(f"  {zone.capitalize()}: {percentage:.1f}%")
            
            # Visualize the route
            router.visualize_route(route)
            print(f"Route visualization saved for {location[2]}")
        else:
            print(f"No viable route found from {location[2]}")
        
        print("\n" + "=" * 50)

if __name__ == "__main__":
    test_evacuation_routes()
# def retrain_and_test():
#     # Initialize router with GPU support
#     router = MumbaiEvacuationRouter(load_model=False, use_gpu=True)
    
#     print("\nStarting training with enhanced rewards...")
#     print("This may take several minutes...")
    
#     # Train the model with more episodes
#     metrics = router.train_rl(num_episodes=1000)  # Increased episodes for better learning
    
#     if metrics and isinstance(metrics, dict):  # Check if metrics is a dictionary
#         print("\nTraining Results:")
#         if 'rewards' in metrics:
#             avg_reward = sum(metrics['rewards'][-100:]) / 100  # Average of last 100 episodes
#             print(f"Average Reward (last 100 episodes): {avg_reward:.2f}")
#         if 'success_rate' in metrics:
#             success_rate = sum(metrics['success_rate'][-100:]) / 100
#             print(f"Success Rate (last 100 episodes): {success_rate:.2f}%")
    
#     # Test points - Added more diverse test locations
#     test_points = [
#         (19.0596, 72.8295, "Dharavi_Center"),
#         (19.0586, 72.8285, "Dharavi_South"),
#         (19.1890, 72.9754, "Thane"),
#         (19.0330, 72.8457, "Worli"),  # Added safer destination
#         (19.2179, 72.8348, "Dahisar")  # Added known safe zone
#     ]
    
#     for start in test_points:
#         try:
#             print(f"\nTesting evacuation from {start[2]}")
#             print("-" * 50)
            
#             route, stats = router.find_best_evacuation_route(start)
#             if route and stats:
#                 print("\nRoute found!")
#                 print(f"Total nodes: {len(route)}")
#                 if 'zone_percentages' in stats:
#                     print("Zone distribution:")
#                     for zone, percentage in stats['zone_percentages'].items():
#                         print(f"  {zone.capitalize()}: {percentage:.1f}%")
                
#                 # Save visualization with unique name
#                 output_file = f"route_{start[2].lower()}.html"
#                 router.visualize_route(route)
#                 print(f"Route visualization saved as {output_file}")
#             else:
#                 print("Failed to find viable route!")
                
#         except Exception as e:
#             print(f"Error testing {start[2]}: {str(e)}")
#             import traceback
#             traceback.print_exc()  # Print full stack trace for debugging
        
#         print("=" * 50)

# if __name__ == "__main__":
#     retrain_and_test()