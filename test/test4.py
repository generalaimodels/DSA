
from typing import Dict, List, Union, Optional, Tuple, Set, Any, Callable
from collections import defaultdict, deque
from heapq import heappush, heappop
import time
import threading
from datetime import datetime, timedelta
import bisect
from test2 import AdvancedDSAAlgorithms
class RealTimeAdvancedDSA:
    
    def __init__(self):
        self.algorithms = AdvancedDSAAlgorithms()
        self.active_sessions = {}
        self.data_streams = defaultdict(deque)
        self.real_time_cache = {}
        
    def real_time_leaderboard_system(self, score_updates: List[Tuple[str, int]]) -> Dict[str, Union[List[Tuple[str, int]], int]]:
        max_heap = []
        user_scores = defaultdict(int)
        
        for user, score in score_updates:
            user_scores[user] += score
            heappush(max_heap, (-user_scores[user], user))
        
        leaderboard = []
        seen = set()
        temp_heap = []
        
        while max_heap and len(leaderboard) < 10:
            neg_score, user = heappop(max_heap)
            if user not in seen:
                leaderboard.append((user, -neg_score))
                seen.add(user)
            temp_heap.append((neg_score, user))
        
        for item in temp_heap:
            heappush(max_heap, item)
        
        return {"leaderboard": leaderboard, "total_users": len(user_scores)}
    
    def real_time_autocomplete_system(self, search_history: List[str], current_query: str) -> List[str]:
        class TrieNode:
            def __init__(self):
                self.children = {}
                self.is_end = False
                self.frequency = 0
                self.sentences = []
        
        root = TrieNode()
        
        for sentence in search_history:
            node = root
            for char in sentence:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
                if sentence not in node.sentences:
                    node.sentences.append(sentence)
                if len(node.sentences) > 3:
                    node.sentences.sort(key=lambda x: search_history.count(x), reverse=True)
                    node.sentences = node.sentences[:3]
            node.is_end = True
            node.frequency += 1
        
        node = root
        for char in current_query:
            if char not in node.children:
                return []
            node = node.children[char]
        
        return sorted(node.sentences, key=lambda x: search_history.count(x), reverse=True)[:3]
    
    def real_time_fraud_detection(self, transactions: List[Dict[str, Union[str, float, int]]]) -> Dict[str, List[Dict[str, Any]]]:
        user_transactions = defaultdict(list)
        fraud_alerts = []
        suspicious_patterns = []
        
        for transaction in transactions:
            user_id = transaction["user_id"]
            amount = transaction["amount"]
            timestamp = transaction["timestamp"]
            location = transaction["location"]
            
            user_transactions[user_id].append(transaction)
            recent_transactions = [t for t in user_transactions[user_id] 
                                 if timestamp - t["timestamp"] <= 3600]
            
            if len(recent_transactions) > 5:
                fraud_alerts.append({
                    "user_id": user_id,
                    "reason": "Too many transactions in short time",
                    "transaction": transaction
                })
            
            if len(recent_transactions) >= 2:
                locations = set(t["location"] for t in recent_transactions[-2:])
                if len(locations) > 1:
                    suspicious_patterns.append({
                        "user_id": user_id,
                        "reason": "Multiple locations",
                        "transaction": transaction
                    })
            
            avg_amount = sum(t["amount"] for t in user_transactions[user_id]) / len(user_transactions[user_id])
            if amount > avg_amount * 3:
                fraud_alerts.append({
                    "user_id": user_id,
                    "reason": "Unusual amount",
                    "transaction": transaction
                })
        
        return {"fraud_alerts": fraud_alerts, "suspicious_patterns": suspicious_patterns}
    
    def real_time_recommendation_engine(self, user_interactions: List[Tuple[str, str, float]], target_user: str) -> List[Tuple[str, float]]:
        user_item_matrix = defaultdict(dict)
        item_users = defaultdict(set)
        
        for user, item, rating in user_interactions:
            user_item_matrix[user][item] = rating
            item_users[item].add(user)
        
        target_items = set(user_item_matrix[target_user].keys())
        recommendations = defaultdict(float)
        
        for item in target_items:
            similar_users = item_users[item] - {target_user}
            
            for similar_user in similar_users:
                similarity = len(set(user_item_matrix[target_user].keys()) & 
                               set(user_item_matrix[similar_user].keys())) / \
                            len(set(user_item_matrix[target_user].keys()) | 
                               set(user_item_matrix[similar_user].keys()))
                
                for other_item, rating in user_item_matrix[similar_user].items():
                    if other_item not in target_items:
                        recommendations[other_item] += similarity * rating
        
        return sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:10]
    
    def real_time_traffic_routing(self, road_network: Dict[str, List[Tuple[str, int]]], traffic_updates: List[Tuple[str, str, int]]) -> Dict[str, Dict[str, Union[List[str], int]]]:
        current_traffic = defaultdict(dict)
        
        for start, end, delay in traffic_updates:
            current_traffic[start][end] = delay
        
        adjusted_network = defaultdict(list)
        for node, neighbors in road_network.items():
            for neighbor, base_time in neighbors:
                traffic_delay = current_traffic[node].get(neighbor, 0)
                adjusted_network[node].append((neighbor, base_time + traffic_delay))
        
        def find_shortest_path(start: str, end: str) -> Tuple[List[str], int]:
            distances = defaultdict(lambda: float('inf'))
            distances[start] = 0
            previous = {}
            heap = [(0, start)]
            visited = set()
            
            while heap:
                current_dist, current_node = heappop(heap)
                
                if current_node in visited:
                    continue
                    
                visited.add(current_node)
                
                if current_node == end:
                    path = []
                    node = end
                    while node in previous:
                        path.append(node)
                        node = previous[node]
                    path.append(start)
                    return path[::-1], current_dist
                
                for neighbor, weight in adjusted_network[current_node]:
                    distance = current_dist + weight
                    if distance < distances[neighbor]:
                        distances[neighbor] = distance
                        previous[neighbor] = current_node
                        heappush(heap, (distance, neighbor))
            
            return [], float('inf')
        
        routes = {}
        major_intersections = list(road_network.keys())[:5]
        
        for i, start in enumerate(major_intersections):
            for end in major_intersections[i+1:]:
                path, time = find_shortest_path(start, end)
                routes[f"{start}_to_{end}"] = {"path": path, "time": time}
        
        return routes
    
    def real_time_social_media_feed(self, posts: List[Dict[str, Union[str, int, List[str]]]], user_id: str, user_connections: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        user_interests = defaultdict(int)
        feed_posts = []
        
        for post in posts:
            post_user = post["user_id"]
            content = post["content"]
            tags = post.get("tags", [])
            timestamp = post["timestamp"]
            
            relevance_score = 0
            
            if post_user in user_connections.get(user_id, []):
                relevance_score += 10
            
            if post_user == user_id:
                relevance_score += 5
            
            for tag in tags:
                user_interests[tag] += 1
                relevance_score += user_interests[tag] * 2
            
            engagement_score = post.get("likes", 0) + post.get("comments", 0) * 2
            relevance_score += engagement_score * 0.1
            
            time_decay = max(0, 100 - (time.time() - timestamp) / 3600)
            relevance_score += time_decay
            
            feed_posts.append({
                "post": post,
                "relevance_score": relevance_score
            })
        
        feed_posts.sort(key=lambda x: x["relevance_score"], reverse=True)
        return [item["post"] for item in feed_posts[:20]]
    
    def real_time_load_balancer(self, servers: List[Dict[str, Union[str, int]]], requests: List[Dict[str, Union[str, int]]]) -> Dict[str, Union[Dict[str, int], List[str]]]:
        server_loads = {server["id"]: 0 for server in servers}
        server_capacities = {server["id"]: server["capacity"] for server in servers}
        request_assignments = []
        rejected_requests = []
        
        def get_least_loaded_server() -> Optional[str]:
            available_servers = [(load, server_id) for server_id, load in server_loads.items() 
                               if load < server_capacities[server_id]]
            
            if not available_servers:
                return None
            
            return min(available_servers)[1]
        
        for request in requests:
            request_id = request["id"]
            resource_need = request["resource_requirement"]
            
            best_server = get_least_loaded_server()
            
            if best_server and server_loads[best_server] + resource_need <= server_capacities[best_server]:
                server_loads[best_server] += resource_need
                request_assignments.append(f"Request {request_id} -> Server {best_server}")
            else:
                rejected_requests.append(f"Request {request_id}")
        
        return {
            "server_loads": server_loads,
            "assignments": request_assignments,
            "rejected": rejected_requests
        }
    
    def real_time_stock_trading_signals(self, price_data: List[Tuple[str, float, int]], window_size: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        stock_data = defaultdict(list)
        signals = defaultdict(list)
        
        for symbol, price, timestamp in price_data:
            stock_data[symbol].append((price, timestamp))
            
            if len(stock_data[symbol]) >= window_size:
                recent_prices = [p for p, _ in stock_data[symbol][-window_size:]]
                
                sma = sum(recent_prices) / len(recent_prices)
                current_price = recent_prices[-1]
                
                if len(recent_prices) >= 2:
                    price_change = (recent_prices[-1] - recent_prices[-2]) / recent_prices[-2] * 100
                    
                    if current_price > sma * 1.02 and price_change > 2:
                        signals[symbol].append({
                            "signal": "BUY",
                            "price": current_price,
                            "confidence": min(95, 50 + price_change * 5),
                            "timestamp": timestamp
                        })
                    elif current_price < sma * 0.98 and price_change < -2:
                        signals[symbol].append({
                            "signal": "SELL",
                            "price": current_price,
                            "confidence": min(95, 50 + abs(price_change) * 5),
                            "timestamp": timestamp
                        })
                
                volatility = max(recent_prices) - min(recent_prices)
                if volatility > sma * 0.1:
                    signals[symbol].append({
                        "signal": "HIGH_VOLATILITY",
                        "price": current_price,
                        "volatility": volatility,
                        "timestamp": timestamp
                    })
        
        return dict(signals)
    
    def real_time_chat_message_router(self, messages: List[Dict[str, Union[str, int]]], user_rooms: Dict[str, List[str]]) -> Dict[str, List[Dict[str, Any]]]:
        room_messages = defaultdict(list)
        user_deliveries = defaultdict(list)
        
        for message in messages:
            sender = message["sender"]
            content = message["content"]
            room = message["room"]
            timestamp = message["timestamp"]
            priority = message.get("priority", 1)
            
            recipients = user_rooms.get(room, [])
            
            for recipient in recipients:
                if recipient != sender:
                    user_deliveries[recipient].append({
                        "message": content,
                        "sender": sender,
                        "room": room,
                        "timestamp": timestamp,
                        "priority": priority
                    })
            
            room_messages[room].append({
                "sender": sender,
                "content": content,
                "timestamp": timestamp,
                "recipients_count": len(recipients) - 1
            })
        
        for user in user_deliveries:
            user_deliveries[user].sort(key=lambda x: (-x["priority"], x["timestamp"]))
        
        return {
            "room_messages": dict(room_messages),
            "user_deliveries": dict(user_deliveries)
        }
    
    def real_time_game_matchmaking(self, players: List[Dict[str, Union[str, int]]], game_modes: Dict[str, Dict[str, int]]) -> Dict[str, List[List[Dict[str, Any]]]]:
        mode_queues = defaultdict(list)
        matches = defaultdict(list)
        
        for player in players:
            player_id = player["id"]
            skill_level = player["skill_level"]
            preferred_mode = player["mode"]
            wait_time = player.get("wait_time", 0)
            
            mode_queues[preferred_mode].append({
                "player": player,
                "adjusted_skill": skill_level + (wait_time // 30),
                "priority": wait_time
            })
        
        for mode, queue in mode_queues.items():
            if mode not in game_modes:
                continue
                
            required_players = game_modes[mode]["players_per_match"]
            skill_tolerance = game_modes[mode]["skill_tolerance"]
            
            queue.sort(key=lambda x: (-x["priority"], x["adjusted_skill"]))
            
            while len(queue) >= required_players:
                potential_match = []
                base_player = queue.pop(0)
                potential_match.append(base_player)
                base_skill = base_player["adjusted_skill"]
                
                remaining_queue = []
                for queued_player in queue:
                    if len(potential_match) < required_players:
                        skill_diff = abs(queued_player["adjusted_skill"] - base_skill)
                        if skill_diff <= skill_tolerance:
                            potential_match.append(queued_player)
                        else:
                            remaining_queue.append(queued_player)
                    else:
                        remaining_queue.append(queued_player)
                
                if len(potential_match) == required_players:
                    matches[mode].append([p["player"] for p in potential_match])
                    queue = remaining_queue
                else:
                    queue = [base_player] + remaining_queue
                    break
        
        return dict(matches)
    
    def real_time_system_monitoring(self, metrics: List[Dict[str, Union[str, float, int]]], thresholds: Dict[str, float]) -> Dict[str, List[Dict[str, Any]]]:
        alerts = []
        trends = defaultdict(list)
        anomalies = []
        
        metric_history = defaultdict(deque)
        
        for metric in metrics:
            metric_name = metric["name"]
            value = metric["value"]
            timestamp = metric["timestamp"]
            host = metric["host"]
            
            metric_history[f"{host}_{metric_name}"].append((value, timestamp))
            
            if len(metric_history[f"{host}_{metric_name}"]) > 100:
                metric_history[f"{host}_{metric_name}"].popleft()
            
            if metric_name in thresholds and value > thresholds[metric_name]:
                alerts.append({
                    "type": "THRESHOLD_EXCEEDED",
                    "metric": metric_name,
                    "host": host,
                    "value": value,
                    "threshold": thresholds[metric_name],
                    "timestamp": timestamp
                })
            
            recent_values = [v for v, _ in list(metric_history[f"{host}_{metric_name}"])[-10:]]
            if len(recent_values) >= 5:
                avg = sum(recent_values) / len(recent_values)
                std_dev = (sum((x - avg) ** 2 for x in recent_values) / len(recent_values)) ** 0.5
                
                if abs(value - avg) > 2 * std_dev and std_dev > 0:
                    anomalies.append({
                        "type": "STATISTICAL_ANOMALY",
                        "metric": metric_name,
                        "host": host,
                        "value": value,
                        "expected_range": (avg - 2*std_dev, avg + 2*std_dev),
                        "timestamp": timestamp
                    })
                
                if len(recent_values) >= 3:
                    if all(recent_values[i] > recent_values[i-1] for i in range(1, len(recent_values))):
                        trends[f"{host}_{metric_name}"].append({
                            "trend": "INCREASING",
                            "duration": len(recent_values),
                            "rate": (recent_values[-1] - recent_values[0]) / len(recent_values),
                            "timestamp": timestamp
                        })
                    elif all(recent_values[i] < recent_values[i-1] for i in range(1, len(recent_values))):
                        trends[f"{host}_{metric_name}"].append({
                            "trend": "DECREASING",
                            "duration": len(recent_values),
                            "rate": (recent_values[0] - recent_values[-1]) / len(recent_values),
                            "timestamp": timestamp
                        })
        
        return {
            "alerts": alerts,
            "trends": dict(trends),
            "anomalies": anomalies
        }
