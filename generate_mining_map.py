import pandas as pd
import googlemaps
import folium
import os
import sys
import time
import math
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
# TODO: Replace with your actual Google Maps API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("Warning: GOOGLE_API_KEY environment variable not set. Please create a .env file.")
INPUT_FILE = "Mining_Locations .xlsx"
OUTPUT_MATRIX_FILE = "driving_distance_matrix.csv"
OUTPUT_MAP_FILE = "mining_map.html"
UPDATED_DATA_FILE = "Mining_Locations_Geocoded.csv"

def load_data(filepath):
    # Check if geocoded data exists first
    if os.path.exists(UPDATED_DATA_FILE):
        print(f"Loading cached geocoded data from {UPDATED_DATA_FILE}...")
        df = pd.read_csv(UPDATED_DATA_FILE)
        return df

    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found.")
        sys.exit(1)

    print(f"Loading data from {filepath}...")
    try:
        df = pd.read_excel(filepath)
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        sys.exit(1)
    
    # Clean column names (strip whitespace)
    df.columns = df.columns.str.strip()
    
    # Drop completely empty rows or rows where Mining Project is NaN
    df = df.dropna(subset=['Mining Project'])
    
    # Filter out the "Quebec" header row if it exists (heuristic: Latitude == 'Quebec')
    df = df[df['Latitude'] != 'Quebec']
    
    return df

def geocode_missing_locations(df, api_key):
    """
    Geocodes locations that have missing Latitude/Longitude.
    """
    if api_key == "YOUR_GOOGLE_API_KEY_HERE":
        print("\n[WARNING] Google Maps API Key not set. Cannot geocode missing locations.")
        return df

    try:
        gmaps = googlemaps.Client(key=api_key)
    except Exception as e:
        print(f"Error initializing Google Maps client for geocoding: {e}")
        return df

    print("\nChecking for missing coordinates...")
    
    # Ensure columns exist
    if 'Latitude' not in df.columns: df['Latitude'] = None
    if 'Longitude' not in df.columns: df['Longitude'] = None
    
    # Convert to numeric, coercing errors to NaN
    df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
    df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')

    missing_mask = df['Latitude'].isna() | df['Longitude'].isna()
    missing_count = missing_mask.sum()
    
    if missing_count == 0:
        print("All locations have coordinates.")
        return df
    
    print(f"Geocoding {missing_count} locations...")
    
    for idx, row in df[missing_mask].iterrows():
        # Construct address
        # Try: Mining Project, Location, Postal Code
        parts = [
            str(row['Mining Project']) if pd.notna(row['Mining Project']) else "",
            str(row['Location']) if pd.notna(row.get('Location', '')) else "",
            str(row['Postal Code']) if pd.notna(row.get('Postal Code', '')) else ""
        ]
        address = ", ".join([p for p in parts if p]).strip()
        
        # Fallback to just Location + Postal Code if Project name might confuse it?
        # Usually project name helps if it's a known mine.
        
        print(f"  Geocoding: {address}")
        
        try:
            geocode_result = gmaps.geocode(address)
            
            if geocode_result:
                location = geocode_result[0]['geometry']['location']
                df.at[idx, 'Latitude'] = location['lat']
                df.at[idx, 'Longitude'] = location['lng']
                print(f"    -> Found: {location['lat']}, {location['lng']}")
            else:
                # Try simpler address (Location + Postal Code)
                simple_address = ", ".join([p for p in parts[1:] if p]).strip()
                if simple_address and simple_address != address:
                    print(f"    -> Retrying with: {simple_address}")
                    geocode_result = gmaps.geocode(simple_address)
                    if geocode_result:
                        location = geocode_result[0]['geometry']['location']
                        df.at[idx, 'Latitude'] = location['lat']
                        df.at[idx, 'Longitude'] = location['lng']
                        print(f"    -> Found: {location['lat']}, {location['lng']}")
                    else:
                        print("    -> Not found.")
                else:
                    print("    -> Not found.")
                    
        except Exception as e:
            print(f"    -> Error: {e}")
            time.sleep(1)
            
    # Save the geocoded data
    df.to_csv(UPDATED_DATA_FILE, index=False)
    print(f"Geocoded data saved to {UPDATED_DATA_FILE}")
    
    return df

def calculate_distance_matrix(df, api_key):
    """
    Calculates driving distance matrix using Google Maps API.
    Returns a DataFrame with the distance matrix (in kilometers).
    """
    if api_key == "YOUR_GOOGLE_API_KEY_HERE":
        print("\n[WARNING] Google Maps API Key is not set. Skipping distance calculation.")
        return None

    try:
        gmaps = googlemaps.Client(key=api_key)
    except Exception as e:
        print(f"Error initializing Google Maps client: {e}")
        return None

    # Filter only valid locations
    df_valid = df.dropna(subset=['Latitude', 'Longitude']).copy()
    if len(df_valid) == 0:
        print("No valid locations to calculate distances for.")
        return None

    locations = df_valid[['Latitude', 'Longitude']].to_records(index=False)
    coords = [(lat, lon) for lat, lon in locations]
    names = df_valid['Mining Project'].tolist()
    
    n = len(coords)
    distance_matrix = pd.DataFrame(index=names, columns=names)
    
    print(f"\nCalculating distance matrix for {n} locations...")
    
    chunk_size = 10
    
    for i in range(0, n, chunk_size):
        origin_chunk = coords[i:i+chunk_size]
        origin_names = names[i:i+chunk_size]
        
        for j in range(0, n, chunk_size):
            dest_chunk = coords[j:j+chunk_size]
            dest_names = names[j:j+chunk_size]
            
            # Avoid re-calculating if not needed, but API is efficient with chunks
            print(f"  Requesting distances from indices {i}-{i+len(origin_chunk)-1} to {j}-{j+len(dest_chunk)-1}...")
            
            try:
                result = gmaps.distance_matrix(origins=origin_chunk, destinations=dest_chunk, mode="driving")
                
                if result['status'] != 'OK':
                    print(f"  API Error: {result.get('error_message', result['status'])}")
                    continue
                
                for r_idx, row in enumerate(result['rows']):
                    origin_name = origin_names[r_idx]
                    for c_idx, element in enumerate(row['elements']):
                        dest_name = dest_names[c_idx]
                        if element['status'] == 'OK':
                            distance_km = element['distance']['value'] / 1000.0
                            distance_matrix.at[origin_name, dest_name] = distance_km
                        else:
                            distance_matrix.at[origin_name, dest_name] = None
                            
            except Exception as e:
                print(f"  Error during API request: {e}")
                time.sleep(1)

    print("Distance matrix calculation complete.")
    return distance_matrix

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + \
        math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * \
        math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def generate_map(df, matrix=None):
    df_valid = df.dropna(subset=['Latitude', 'Longitude'])
    if len(df_valid) == 0:
        print("No valid locations to map.")
        return

    print("\nGenerating interactive map...")
    
    center_lat = df_valid['Latitude'].mean()
    center_lon = df_valid['Longitude'].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=5)
    
    locations = []
    for idx, row in df_valid.iterrows():
        locations.append({
            'name': row['Mining Project'],
            'lat': row['Latitude'],
            'lon': row['Longitude']
        })
        
        folium.Marker(
            [row['Latitude'], row['Longitude']],
            popup=f"{row['Mining Project']}",
            tooltip=row['Mining Project']
        ).add_to(m)
    
    print("  Drawing connections to nearest neighbors...")
    
    for i, loc1 in enumerate(locations):
        distances = []
        for j, loc2 in enumerate(locations):
            if i == j:
                continue
            
            dist = 0
            if matrix is not None and loc1['name'] in matrix.index and loc2['name'] in matrix.columns:
                val = matrix.at[loc1['name'], loc2['name']]
                
                # Handle duplicate entries returning Series or DataFrame
                if isinstance(val, (pd.Series, pd.DataFrame)):
                     # If it's a DataFrame/Series, flatten and take the first non-null value if possible
                     vals = val.values.flatten()
                     # Find first non-nan
                     found_dist = False
                     for v in vals:
                         if pd.notna(v):
                             val = v
                             found_dist = True
                             break
                     if not found_dist:
                         val = None

                if pd.notna(val):
                    dist = float(val)
                else:
                    dist = float('inf')
            else:
                dist = haversine_distance(loc1['lat'], loc1['lon'], loc2['lat'], loc2['lon'])
            
            distances.append((dist, loc2))
        
        distances.sort(key=lambda x: x[0])
        nearest = distances[:3]
        
        for dist, loc2 in nearest:
            if dist == float('inf'):
                continue
                
            folium.PolyLine(
                locations=[[loc1['lat'], loc1['lon']], [loc2['lat'], loc2['lon']]],
                weight=1,
                color='blue',
                opacity=0.5,
                tooltip=f"Dist: {dist:.2f} km"
            ).add_to(m)

    m.save(OUTPUT_MAP_FILE)
    print(f"Map saved to {OUTPUT_MAP_FILE}")

def main():
    # 1. Load Data
    df = load_data(INPUT_FILE)
    print(f"Loaded {len(df)} rows.")
    
    # 2. Geocode if needed
    df = geocode_missing_locations(df, GOOGLE_API_KEY)
    
    # 3. Calculate Distance Matrix
    matrix = calculate_distance_matrix(df, GOOGLE_API_KEY)
    
    if matrix is not None:
        matrix.to_csv(OUTPUT_MATRIX_FILE)
        print(f"Distance matrix saved to {OUTPUT_MATRIX_FILE}")
        
    # 4. Generate Map
    generate_map(df, matrix)

if __name__ == "__main__":
    main()
