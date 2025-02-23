import osmnx as ox
import networkx as nx
import geopandas as gpd
from shapely.geometry import Point, LineString
import pandas as pd
import folium
from datetime import datetime
import requests
from math import sqrt
import random
import torch

class MumbaiEvacuationRouter:
    def __init__(self, buffer_radius=2000, load_model=True, use_gpu=True):
        self.danger_zones = {
            'red': [
                (19.0760, 72.8777, "Bandra"),
                (19.1890, 72.9754, "Thane"),
                (19.2335, 73.1305, "Kalyan"),
                (19.0596, 72.8295, "Dharavi"),
            ],
            'yellow': [
                (19.0330, 72.8457, "Worli"),
                (19.2183, 72.9781, "Mulund"),
                (19.1136, 72.8697, "Andheri"),
                (19.0895, 72.8656, "Kurla"),
                (19.1647, 72.8470, "Borivali"),
            ]
        }
        
        self.buffer_radius = buffer_radius
        self.G = None
        self.zone_geometries = {
            'red': [],
            'yellow': []
        }
        self.blue_zones = []
        self.major_roads = set()
        self.essential_minor_roads = set()
        self.q_table = {}  # For RL
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.1
        self.model_path = 'evacuation_model.pth'
        
        # Initialize device before anything else
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        print(f"\nUsing device: {self.device}")
        
        # Initialize tensors on the correct device
        self.learning_rate = torch.tensor(0.1, device=self.device)
        self.discount_factor = torch.tensor(0.9, device=self.device)
        self.epsilon = torch.tensor(0.1, device=self.device)
        
        print(f"\n=== Initializing Mumbai Metropolitan Region Evacuation Router ===")
        self._initialize_network()
        self._verify_graph_connectivity()
        self._detect_safe_zones()
        
        if load_model:
            self._load_model()
        
        self.q_table = {}
        self.learning_rate = torch.tensor(0.1, device=self.device)
        self.discount_factor = torch.tensor(0.9, device=self.device)
        self.epsilon = torch.tensor(0.1, device=self.device)

    def _initialize_network(self):
        """Download and prepare the Mumbai road network with major roads identification"""
        print("\nDownloading road network for Mumbai Metropolitan Region...")
        
        try:
            custom_filter = (
                '["highway"~"motorway|trunk|primary|secondary|'
                'motorway_link|trunk_link|primary_link|secondary_link"]'
            )
            self.G = ox.graph_from_place(
                "Mumbai, Maharashtra, India",
                network_type='drive',
                custom_filter=custom_filter
            )
            
            for u, v, data in self.G.edges(data=True):
                if data.get('highway') in ['motorway', 'trunk', 'primary']:
                    self.major_roads.add((u, v))
            
            print(f"Identified {len(self.major_roads)} major road segments")
            
            self._identify_essential_minor_roads()
            
        except Exception as e:
            print(f"Error downloading network: {e}")
            return
        
        print("\nCreating danger zone buffers...")
        for zone_type in ['red', 'yellow']:
            for coord in self.danger_zones.get(zone_type, []):
                point = Point(coord[1], coord[0])
                buffer_zone = point.buffer(self.buffer_radius / 111000)
                self.zone_geometries[zone_type].append({
                    'geometry': buffer_zone,
                    'name': coord[2]
                })
                print(f"Created {zone_type} zone buffer for {coord[2]}")
        
        self._classify_roads()

    def _detect_safe_zones(self):
        """Initialize safe zones"""
        print("\nInitializing safe zones...")
        
        self.blue_zones = [
            (19.1741, 72.9519, "Powai"),
            (19.0282, 72.8565, "Malabar Hill"),
            (19.2179, 72.8348, "Dahisar"),
            (19.0607, 72.9926, "Vikhroli"),
            (19.1538, 72.9246, "Mulund West")
        ]
        
        filtered_blue_zones = []
        for safe_zone in self.blue_zones:
            is_safe = True
            point = Point(safe_zone[1], safe_zone[0])
            
            for zone in self.zone_geometries['red']:
                if zone['geometry'].contains(point) or zone['geometry'].distance(point) < 0.01:
                    is_safe = False
                    break
                    
            if is_safe:
                for zone in self.zone_geometries['yellow']:
                    if zone['geometry'].contains(point) or zone['geometry'].distance(point) < 0.01:
                        is_safe = False
                        break
            
            if is_safe:
                filtered_blue_zones.append(safe_zone)
        
        self.blue_zones = filtered_blue_zones
        print(f"Identified {len(self.blue_zones)} safe zones:")
        for zone in self.blue_zones:
            print(f"  - {zone[2]} at ({zone[0]:.4f}, {zone[1]:.4f})")

    def _classify_roads(self):
        """Classify Mumbai road segments based on danger zones"""
        print("\nClassifying Mumbai road segments...")
        
        road_classifications = {
            'red': 0,
            'yellow': 0,
            'blue': 0
        }
        
        for u, v, key, data in self.G.edges(keys=True, data=True):
            start = Point(self.G.nodes[u]['x'], self.G.nodes[u]['y'])
            end = Point(self.G.nodes[v]['x'], self.G.nodes[v]['y'])
            road = LineString([start, end])
            
            in_red_zone = False
            in_yellow_zone = False
            
            for zone in self.zone_geometries['red']:
                if zone['geometry'].intersects(road):
                    in_red_zone = True
                    self.G.edges[u, v, key]['zone_name'] = zone['name']
                    break
                    
            if not in_red_zone:
                for zone in self.zone_geometries['yellow']:
                    if zone['geometry'].intersects(road):
                        in_yellow_zone = True
                        self.G.edges[u, v, key]['zone_name'] = zone['name']
                        break
            
            if in_red_zone:
                self.G.edges[u, v, key]['zone'] = 'red'
                self.G.edges[u, v, key]['weight'] = data.get('length', 1) * 100
                road_classifications['red'] += 1
            elif in_yellow_zone:
                self.G.edges[u, v, key]['zone'] = 'yellow'
                self.G.edges[u, v, key]['weight'] = data.get('length', 1) * 10
                road_classifications['yellow'] += 1
            else:
                self.G.edges[u, v, key]['zone'] = 'blue'
                self.G.edges[u, v, key]['weight'] = data.get('length', 1)
                road_classifications['blue'] += 1
        
        print("\nMumbai Road Classification Summary:")
        for zone, count in road_classifications.items():
            print(f"{zone.capitalize()} roads: {count}")

    def _identify_essential_minor_roads(self):
        """Identify minor roads that are critical for evacuation"""
        for blue_zone in self.blue_zones:
            end_node = ox.nearest_nodes(
                self.G, 
                blue_zone[1],
                blue_zone[0]
            )
            nearby_nodes = set()
            for node in self.G.nodes():
                dist = sqrt(
                    (self.G.nodes[node]['y'] - blue_zone[0])**2 + 
                    (self.G.nodes[node]['x'] - blue_zone[1])**2
                )
                if dist < 0.01:  # Approximately 1km
                    nearby_nodes.add(node)
            
            for u, v, data in self.G.edges(data=True):
                if u in nearby_nodes or v in nearby_nodes:
                    if (u, v) not in self.major_roads:
                        self.essential_minor_roads.add((u, v))

    def get_state(self, node):
        """Convert node position and surrounding conditions to state"""
        x, y = self.G.nodes[node]['x'], self.G.nodes[node]['y']
        zone = self._get_node_zone(node)
        nearby_major = any(
            node in edge for edge in self.major_roads
        )
        return (x, y, zone, nearby_major)

    def _get_node_zone(self, node):
        """Get the zone type for a node"""
        point = Point(
            self.G.nodes[node]['x'], 
            self.G.nodes[node]['y']
        )
        return self._check_point_safety(point, "", "")

    def get_reward(self, current_node, next_node, end_node):
        """Calculate reward with improved metrics"""
        reward = torch.tensor(0.0, device=self.device)
        
        # Zone-based rewards (adjusted)
        next_zone = self._get_node_zone(next_node)
        if next_zone == 'red':
            reward -= torch.tensor(500.0, device=self.device)
        elif next_zone == 'yellow':
            reward -= torch.tensor(50.0, device=self.device)
        else:
            reward += torch.tensor(10.0, device=self.device)
        
        if (current_node, next_node) in self.major_roads:
            reward += torch.tensor(20.0, device=self.device)
        elif (current_node, next_node) in self.essential_minor_roads:
            reward += torch.tensor(10.0, device=self.device)
        
        current_dist = self._euclidean_distance(current_node, end_node)
        next_dist = self._euclidean_distance(next_node, end_node)
        progress = current_dist - next_dist
        progress_reward = progress * torch.tensor(50.0, device=self.device)
        reward += progress_reward
        
        if next_node == end_node:
            if self._get_node_zone(end_node) == 'blue':
                reward += torch.tensor(1000.0, device=self.device)
        
        reward -= torch.tensor(1.0, device=self.device)
        return reward

    def _euclidean_distance(self, node1, node2):
        x1, y1 = self.G.nodes[node1]['x'], self.G.nodes[node1]['y']
        x2, y2 = self.G.nodes[node2]['x'], self.G.nodes[node2]['y']
        return sqrt((x2-x1)**2 + (y2-y1)**2)

    def update_dynamic_route(self, start_node, end_node, current_conditions):
        state = self.get_state(start_node)
        
        if state not in self.q_table:
            self.q_table[state] = {
                neighbor: 0 for neighbor in self.G.neighbors(start_node)
            }
        
        if random.random() < self.epsilon:
            next_node = random.choice(list(self.G.neighbors(start_node)))
        else:
            next_node = max(
                self.q_table[state].items(), 
                key=lambda x: x[1]
            )[0]
        
        reward = self.get_reward(start_node, next_node, end_node)
        next_state = self.get_state(next_node)
        if next_state not in self.q_table:
            self.q_table[next_state] = {
                neighbor: 0 for neighbor in self.G.neighbors(next_node)
            }
        
        max_next_q = max(self.q_table[next_state].values())
        self.q_table[state][next_node] += self.learning_rate * (
            reward + 
            self.discount_factor * max_next_q - 
            self.q_table[state][next_node]
        )
        return next_node

    def find_nearest_safe_towns(self, start_location, num_options=3):
        start_lat, start_lon = start_location[0], start_location[1]
        safe_towns = []

        print(f"\nFinding nearest safe towns from {start_location[2]}...")

        # Calculate distances to all blue zones
        for zone in self.blue_zones:
            dist = sqrt((zone[0] - start_lat)**2 + (zone[1] - start_lon)**2)
            safe_towns.append((zone, dist))

        # Sort by distance and get top options
        safe_towns.sort(key=lambda x: x[1])
        return safe_towns[:num_options]

    def find_best_evacuation_route(self, start_location):
        if self.G is None:
            print("Error: Road network not initialized!")
            return None, None

        # Find nearest safe towns
        safe_options = self.find_nearest_safe_towns(start_location)
        
        if not safe_options:
            print("No safe towns found!")
            return None, None

        print("\nEvaluating routes to nearest safe towns:")
        best_route = None
        best_stats = None
        best_destination = None
        best_score = float('inf')

        for safe_town, distance in safe_options:
            print(f"\nTrying route to {safe_town[2]}")
            route, stats = self.find_evacuation_route(start_location, safe_town)
            
            if route and stats:
                route_score = stats['total_distance'] * (
                    1 + 
                    stats['zone_percentages'].get('red', 0) * 2 + 
                    stats['zone_percentages'].get('yellow', 0)
                ) / 100

                print(f"Route score: {route_score:.2f}")
                
                if route_score < best_score:
                    best_score = route_score
                    best_route = route
                    best_stats = stats
                    best_destination = safe_town

        if best_route:
            print(f"\nBest evacuation route found!")
            print(f"Recommended destination: {best_destination[2]}")
            self._print_route_stats(best_stats)
            return best_route, best_stats
        else:
            print("\nNo viable evacuation routes found!")
            return None, None

    def find_evacuation_route(self, start_location, end_location):
        """Find evacuation route using trained RL model"""
        if self.G is None:
            print("Error: Road network not initialized!")
            return None, None

        start_node = ox.nearest_nodes(
            self.G, 
            start_location[1], 
            start_location[0]
        )
        end_node = ox.nearest_nodes(
            self.G, 
            end_location[1], 
            end_location[0]
        )

        temp_epsilon = self.epsilon
        self.epsilon = 0
        
        route = [start_node]
        current_node = start_node
        
        while current_node != end_node and len(route) < 100:
            next_node = self._select_action(current_node, end_node)
            
            if next_node in route:
                break
                
            route.append(next_node)
            current_node = next_node
        
        self.epsilon = temp_epsilon
        return route, self._analyze_route(route)

    def _analyze_route(self, route):
        """Analyze the route and calculate statistics"""
        total_distance = 0
        zone_counts = {'red': 0, 'yellow': 0, 'blue': 0}
        
        for u, v in zip(route[:-1], route[1:]):
            edge_data = min(
                self.G.get_edge_data(u, v).values(),
                key=lambda x: x.get('weight', float('inf'))
            )
            total_distance += edge_data.get('length', 1)
            zone_counts[edge_data['zone']] += 1
        
        total_segments = sum(zone_counts.values())
        if total_segments == 0:
            return {
                'total_distance': 0,
                'zone_counts': zone_counts,
                'zone_percentages': {zone: 0 for zone in zone_counts}
            }
        
        zone_percentages = {
            zone: (count / total_segments) * 100 
            for zone, count in zone_counts.items()
        }
        
        return {
            'total_distance': total_distance,
            'zone_counts': zone_counts,
            'zone_percentages': zone_percentages
        }

    def _check_point_safety(self, point, name, point_type):
        for zone_type in ['red', 'yellow']:
            for zone in self.zone_geometries[zone_type]:
                if zone['geometry'].contains(point):
                    print(f"{point_type} point {name} is in a {zone_type} zone ({zone['name']})")
                    return zone_type
        print(f"{point_type} point {name} is in a blue zone")
        return 'blue'

    def _print_route_stats(self, stats):
        print(f"Total distance: {stats['total_distance']:.2f} meters")
        for zone, count in stats['zone_counts'].items():
            print(f"{zone.capitalize()} zone segments: {count} ({stats['zone_percentages'][zone]:.2f}%)")

    def visualize_route(self, route):
        center_lat = 19.0760
        center_lon = 72.8777
        route_map = folium.Map(location=[center_lat, center_lon], zoom_start=12)
        
        zone_colors = {
            'red': {'fill': '#ff0000', 'color': '#8b0000', 'name': 'High Risk Zone'},
            'yellow': {'fill': '#ffff00', 'color': '#ffd700', 'name': 'Medium Risk Zone'},
            'blue': {'fill': '#0000ff', 'color': '#000080', 'name': 'Safe Zone'}
        }
        
        for zone_type in ['red', 'yellow']:
            for zone in self.zone_geometries[zone_type]:
                bounds = zone['geometry'].bounds
                radius = self.buffer_radius
                folium.Circle(
                    location=[
                        (bounds[1] + bounds[3]) / 2,
                        (bounds[0] + bounds[2]) / 2
                    ],
                    radius=radius,
                    popup=f"{zone['name']} ({zone_colors[zone_type]['name']})",
                    color=zone_colors[zone_type]['color'],
                    fill=True,
                    fill_color=zone_colors[zone_type]['fill'],
                    fill_opacity=0.2
                ).add_to(route_map)
        
        for zone in self.blue_zones:
            folium.Circle(
                location=[zone[0], zone[1]],
                radius=self.buffer_radius,
                popup=f"{zone[2]} (Safe Zone)",
                color=zone_colors['blue']['color'],
                fill=True,
                fill_color=zone_colors['blue']['fill'],
                fill_opacity=0.2
            ).add_to(route_map)
        
        for i in range(len(route)-1):
            node1, node2 = route[i], route[i+1]
            coord1 = [self.G.nodes[node1]['y'], self.G.nodes[node1]['x']]
            coord2 = [self.G.nodes[node2]['y'], self.G.nodes[node2]['x']]
            
            edge_data = min(
                self.G.get_edge_data(node1, node2).values(),
                key=lambda x: x.get('weight', float('inf'))
            )
            zone_type = edge_data.get('zone', 'blue')
            
            weight = 4
            if zone_type == 'blue':
                weight = 6
            if (node1, node2) in self.major_roads:
                weight += 2
            
            folium.PolyLine(
                locations=[coord1, coord2],
                weight=weight,
                color=zone_colors[zone_type]['color'],
                opacity=0.8,
                popup=f"{zone_type.capitalize()} Zone Segment"
            ).add_to(route_map)
        
        folium.Marker(
            [self.G.nodes[route[0]]['y'], self.G.nodes[route[0]]['x']],
            popup='Start',
            icon=folium.Icon(color='green', icon='info-sign')
        ).add_to(route_map)
        
        folium.Marker(
            [self.G.nodes[route[-1]]['y'], self.G.nodes[route[-1]]['x']],
            popup='End',
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(route_map)
        
        legend_html = '''
        <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; background-color: white; padding: 10px; border: 2px solid grey; border-radius: 5px">
            <p><strong>Legend:</strong></p>
            <p><span style="color: #8b0000;">━━━</span> High Risk Route</p>
            <p><span style="color: #ffd700;">━━━</span> Medium Risk Route</p>
            <p><span style="color: #000080;">━━━</span> Safe Route</p>
            <p>🔴 High Risk Zone</p>
            <p>🟡 Medium Risk Zone</p>
            <p>🔵 Safe Zone</p>
            <p>🟢 Start Point</p>
            <p>❌ End Point</p>
        </div>
        '''
        route_map.get_root().html.add_child(folium.Element(legend_html))
        
        route_map.save('evacuation_route.html')
        print("Route map saved as evacuation_route.html")

    def _load_model(self):
        """Load pre-trained model with device handling"""
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.q_table = {
                k: {
                    node: torch.tensor(val, device=self.device) 
                    for node, val in v.items()
                }
                for k, v in checkpoint['q_table'].items()
            }
            print("Loaded pre-trained evacuation model")
        except FileNotFoundError:
            print("No pre-trained model found. Training new model...")
            self.train_rl()

    def _save_model(self, metrics):
        """Save model with CPU tensors for compatibility"""
        print("\nSaving trained model...")
        q_table_cpu = {
            k: {
                node: val.cpu().item() 
                for node, val in v.items()
            }
            for k, v in self.q_table.items()
        }
        
        torch.save({
            'q_table': q_table_cpu,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }, self.model_path)
        print(f"Model saved to {self.model_path}")

    def _generate_training_points(self, num_points=1000):
        """Generate random training points across Mumbai"""
        mumbai_bounds = {
            'north': 19.2813,
            'south': 18.8921,
            'east': 73.0525,
            'west': 72.7754
        }
        
        training_points = []
        for _ in range(num_points):
            lat = random.uniform(mumbai_bounds['south'], mumbai_bounds['north'])
            lon = random.uniform(mumbai_bounds['west'], mumbai_bounds['east'])
            point = (lat, lon, "training_point")
            if ox.distance.nearest_nodes(self.G, lon, lat):
                training_points.append(point)
        
        return training_points

    def train_rl(self, num_episodes=1000):
        print(f"\nTraining RL routing system on {self.device}...")
        
        metrics = {
            'episodes': [],
            'rewards': [],
            'success_rate': [],
            'red_zone_avoidance': []
        }
        
        training_points = self._generate_training_points()
        
        best_avg_reward = float('-inf')
        best_model = None
        
        for episode in range(num_episodes):
            start_point = random.choice(training_points)
            safe_options = self.find_nearest_safe_towns(start_point, num_options=1)
            
            if not safe_options:
                continue
                
            end_point = safe_options[0][0]
            
            episode_data = self._run_training_episode(start_point, end_point)
            
            if (episode + 1) % 1000 == 0:
                avg_reward = torch.tensor(episode_data['total_reward'], device=self.device)
                print(f"\nEpisode {episode + 1}/{num_episodes}")
                print(f"Average Reward: {avg_reward.item():.2f}")
                
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    best_model = {
                        'q_table': self.q_table,
                        'metrics': metrics,
                        'avg_reward': avg_reward.item(),
                        'timestamp': datetime.now().isoformat()
                    }
    
        if best_avg_reward > 800:
            print(f"\nSaving high-performing model (avg reward: {best_avg_reward:.2f})")
            torch.save(best_model, 'evacuation_model.pth')
            print("Model saved successfully!")
        
        return metrics

    def _run_training_episode(self, start_point, end_point):
        """Run training episode with proper error handling"""
        try:
            start_node = ox.nearest_nodes(self.G, start_point[1], start_point[0])
            end_node = ox.nearest_nodes(self.G, end_point[1], end_point[0])
            
            if not self.G.has_node(start_node) or not self.G.has_node(end_node):
                print(f"Invalid nodes: start={start_node}, end={end_node}")
                return None
            
            current_node = start_node
            route = [current_node]
            total_reward = torch.tensor(0.0, device=self.device)
            entered_red_zone = False
            success = False
            
            while len(route) < 100:
                try:
                    next_node = self._select_action(current_node, end_node)
                    
                    reward = self.get_reward(current_node, next_node, end_node)
                    total_reward += reward
                    
                    if self._get_node_zone(next_node) == 'red':
                        entered_red_zone = True
                    
                    route.append(next_node)
                    current_node = next_node
                    
                    if current_node == end_node:
                        success = True
                        break
                        
                except ValueError as e:
                    print(f"Error selecting action: {e}")
                    break
                    
            return {
                'route': route,
                'total_reward': total_reward.item(),
                'success': success,
                'entered_red_zone': entered_red_zone
            }
            
        except Exception as e:
            print(f"Error in training episode: {e}")
            return None

    def _select_action(self, current_node, end_node):
        state = self.get_state(current_node)
        
        if state not in self.q_table:
            neighbors = list(self.G.neighbors(current_node))
            if not neighbors:
                raise ValueError(f"Node {current_node} has no neighbors")
                
            self.q_table[state] = {
                n: torch.tensor(0.0, device=self.device) 
                for n in neighbors
            }
        
        if torch.rand(1, device=self.device) < self.epsilon:
            return random.choice(list(self.G.neighbors(current_node)))
        else:
            if not self.q_table[state]:
                return random.choice(list(self.G.neighbors(current_node)))
                
            return max(
                self.q_table[state].items(),
                key=lambda x: x[1]
            )[0]

    def _save_model(self, metrics):
        print("\nSaving trained model...")
        torch.save({
            'q_table': self.q_table,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }, self.model_path)
        print(f"Model saved to {self.model_path}")

    def _log_training_progress(self, metrics, episode, total):
        recent = slice(-100, None)
        avg_reward = sum(metrics['rewards'][recent]) / 100
        success_rate = sum(metrics['success_rate'][recent])
        avoid_rate = sum(metrics['red_zone_avoidance'][recent])
        
        print(f"\nEpisode {episode}/{total}")
        print(f"Average Reward: {avg_reward:.2f}")
        print(f"Success Rate: {success_rate}%")
        print(f"Red Zone Avoidance: {avoid_rate}%")
        print(f"Zone Distribution: Red {metrics['zone_distribution']['red']:.1f}%, " 
              f"Yellow {metrics['zone_distribution']['yellow']:.1f}%, "
              f"Blue {metrics['zone_distribution']['blue']:.1f}%")

    def _verify_graph_connectivity(self):
        if not self.G:
            raise ValueError("Graph not initialized")
            
        if len(self.G.nodes()) == 0:
            raise ValueError("Graph has no nodes")
            
        isolated_nodes = list(nx.isolates(self.G))
        if isolated_nodes:
            print(f"Warning: Found {len(isolated_nodes)} isolated nodes")
            self.G.remove_nodes_from(isolated_nodes)
            
        if len(self.G.edges()) == 0:
            raise ValueError("Graph has no edges")
            
        print(f"Graph verification complete: {len(self.G.nodes())} nodes, {len(self.G.edges())} edges")

def main():
    router = MumbaiEvacuationRouter(load_model=False, use_gpu=True)
    router.train_rl(num_episodes=5000)
    
    test_locations = [
        (19.0596, 72.8295, "Dharavi"),
        (19.0760, 72.8777, "Bandra"),
        (19.1136, 72.8697, "Andheri")
    ]
    
    for start in test_locations:
        print(f"\nFinding evacuation route from {start[2]}")
        route, stats = router.find_best_evacuation_route(start)
        if route:
            router.visualize_route(route)
            router._print_route_stats(stats)
        else:
            print(f"No safe route found from {start[2]}")

if __name__ == "__main__":
    main()