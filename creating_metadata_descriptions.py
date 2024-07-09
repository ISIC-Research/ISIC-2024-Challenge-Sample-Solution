import pandas as pd

# Load the CSV file
data = pd.read_csv("Data/train-metadata.csv", low_memory=False)

# Define a function to generate the text description, handling missing values
def generate_description(row):
    age = row['age_approx'] if pd.notna(row['age_approx']) else 'unknown age'
    sex = row['sex'] if pd.notna(row['sex']) else 'unknown sex'
    attribution = row['attribution'] if pd.notna(row['attribution']) else 'unknown treatment center'
    atomic_site = row['anatom_site_general'] if pd.notna(row['anatom_site_general']) else 'unknown site'
    area = row['tbp_lv_areaMM2'] if pd.notna(row['tbp_lv_areaMM2']) else 'unknown'
    diameter = row['clin_size_long_diam_mm'] if pd.notna(row['clin_size_long_diam_mm']) else 'unknown'
    perimeter = row['tbp_lv_perimeterMM'] if pd.notna(row['tbp_lv_perimeterMM']) else 'unknown'
    jaggedness = row['tbp_lv_area_perim_ratio'] if pd.notna(row['tbp_lv_area_perim_ratio']) else 'unknown'
    boarder = row['tbp_lv_norm_border'] if pd.notna(row['tbp_lv_norm_border']) else 'unknown'
    color_variation = row['tbp_lv_norm_color'] if pd.notna(row['tbp_lv_norm_color']) else 'unknown'
    image_modality=row['tbp_tile_type'] if pd.notna(row['tbp_tile_type']) else 'unknown'
    eccentricity=row['tbp_lv_eccentricity'] if pd.notna(row['tbp_lv_eccentricity']) else 'unknown'
    
    description = (f"A {age} {sex} presented to {attribution} with a skin lesion on {atomic_site} that has a {area} area, {diameter} diameter, and {perimeter} perimeter. "
                   f"The lesion has a {jaggedness} border jaggedness ratio, {boarder} border irregularity, and {color_variation} color variance. "
                   f"The lesion has {eccentricity} eccentricity. "
                   f"This image was taken with {image_modality} lighting. ")
    
    return description

# Apply the function to each row in the DataFrame
data['description'] = data.apply(generate_description, axis=1)

# Save the updated DataFrame to a new CSV file
data.to_csv('Data/train_metadata_with_descriptions.csv', index=False)

# Optional: Print the first few rows to check the descriptions
print(data[['description']].head())




# Load the CSV file
data = pd.read_csv("Data/test-metadata.csv", low_memory=False)

# Define a function to generate the text description, handling missing values
def generate_description(row):
    age = row['age_approx'] if pd.notna(row['age_approx']) else 'unknown age'
    sex = row['sex'] if pd.notna(row['sex']) else 'unknown sex'
    attribution = row['attribution'] if pd.notna(row['attribution']) else 'unknown treatment center'
    atomic_site = row['anatom_site_general'] if pd.notna(row['anatom_site_general']) else 'unknown site'
    area = row['tbp_lv_areaMM2'] if pd.notna(row['tbp_lv_areaMM2']) else 'unknown'
    diameter = row['clin_size_long_diam_mm'] if pd.notna(row['clin_size_long_diam_mm']) else 'unknown'
    perimeter = row['tbp_lv_perimeterMM'] if pd.notna(row['tbp_lv_perimeterMM']) else 'unknown'
    jaggedness = row['tbp_lv_area_perim_ratio'] if pd.notna(row['tbp_lv_area_perim_ratio']) else 'unknown'
    boarder = row['tbp_lv_norm_border'] if pd.notna(row['tbp_lv_norm_border']) else 'unknown'
    color_variation = row['tbp_lv_norm_color'] if pd.notna(row['tbp_lv_norm_color']) else 'unknown'
    image_modality=row['tbp_tile_type'] if pd.notna(row['tbp_tile_type']) else 'unknown'
    eccentricity=row['tbp_lv_eccentricity'] if pd.notna(row['tbp_lv_eccentricity']) else 'unknown'
    
    description = (f"A {age} {sex} presented to {attribution} with a skin lesion on {atomic_site} that has a {area} area, {diameter} diameter, and {perimeter} perimeter. "
                   f"The lesion has a {jaggedness} border jaggedness ratio, {boarder} border irregularity, and {color_variation} color variance. "
                   f"The lesion has {eccentricity} eccentricity. "
                   f"This image was taken with {image_modality} lighting. ")
    
    return description

# Apply the function to each row in the DataFrame
data['description'] = data.apply(generate_description, axis=1)

# Save the updated DataFrame to a new CSV file
data.to_csv('Data/test_metadata_with_descriptions.csv', index=False)

# Optional: Print the first few rows to check the descriptions
print(data[['description']].head())