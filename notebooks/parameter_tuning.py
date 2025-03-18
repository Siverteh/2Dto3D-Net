import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
import yaml
import open3d as o3d
from tqdm.notebook import tqdm
import itertools
import time
import datetime
import copy
import json
from pathlib import Path

# Add the project directory to the Python path
sys.path.append('..')

# Import project modules
from src.preprocessing.image_loader import load_image_sequence, resize_images
from src.preprocessing.image_preprocessing import enhance_contrast
from src.feature_extraction.sift_extractor import extract_features_from_image_set
from src.feature_extraction.orb_extractor import extract_distributed_orb_features, extract_orb_features
from src.feature_extraction.feature_matcher import match_image_pairs, geometric_verification
from src.sfm.camera_calibration import estimate_camera_matrix
from src.sfm.pose_estimation import estimate_poses_incremental, force_loop_closure
from src.sfm.triangulation import triangulate_all_points, merge_triangulated_points
from src.sfm.bundle_adjustment import run_global_ba
from src.dense_reconstruction.mvs import process_mvs
from src.dense_reconstruction.point_cloud import process_dense_reconstruction, create_surface_mesh, save_mesh
from src.surface_reconstruction.mesh_generation import process_point_cloud_to_mesh, clean_mesh
from src.surface_reconstruction.texture_mapping import create_textured_mesh_from_point_cloud

from src.visualization.plot_matches import plot_matches, plot_feature_matching_analysis
from src.visualization.point_cloud_visualizer import plot_interactive_point_cloud, create_point_cloud_animation
from src.visualization.camera_visualizer import plot_interactive_camera_poses
from src.visualization.mesh_visualizer import visualize_mesh_o3d, plot_interactive_mesh

# Set up matplotlib for inline display
plt.style.use('default')

# Base configuration (from paste-2.txt)
base_config = {
    'preprocessing': {
        'resize_max_dimension': 1000,
        'enhance_contrast': True
    },
    'features': {
        'method': 'sift',
        'max_features': 50000,
        'use_multiscale': True,
        'use_dense': True,
        'dense_step': 5,
        'contrast_threshold': 0.01,
        'edge_threshold': 8
    },
    'matching': {
        'ratio_threshold': 0.7,
        'geometric_verification': True,
        'min_matches': 16,
        'verification_method': 'fundamental',
        'ransac_threshold': 2.0,
        'cross_check': True,
        'max_epipolar_error': 1.0,
        'confidence': 0.999
    },
    'calibration': {
        'focal_length_factor': 1.3,
        'principal_point': 'center',
        'refine_intrinsics': True
    },
    'sfm': {
        'incremental': True,
        'refine_poses': True,
        'min_triangulation_angle_deg': 3.0,
        'reprojection_error_threshold': 2.0,
        'bundle_adjustment_max_iterations': 100
    },
    'mvs': {
        'min_disparity': 0,
        'num_disparities': 128,
        'block_size': 7,
        'filter_depths': True,
        'consistency_threshold': 0.01,
        'num_source_views': 5
    },
    'point_cloud': {
        'voxel_size': 0.02,
        'nb_neighbors': 30,
        'std_ratio': 1.5,
        'confidence_threshold': 0.8
    },
    'surface': {
        'method': 'poisson',
        'depth': 10,
        'cleanup': True,
        'trim': 7.0
    },
    'visualization': {
        'point_size': 2,
        'camera_size': 6,
        'point_color_method': 'rgb'
    }
}

hyperparameter_grid = {
    'features': {
        'method': ['sift'],  
        'max_features': [10000, 30000, 50000],
        'contrast_threshold': [0.005, 0.01, 0.015]
    },
    'matching': {
        'ratio_threshold': [0.65, 0.75, 0.85],
        'min_matches': [10, 16, 22]
    },
    'calibration': {
        'focal_length_factor': [1.3]
    },
    'sfm': {
        'min_triangulation_angle_deg': [2.0, 3.0, 4.0],  # Lower = more points but less stable depth
        'max_reprojection_error': [3.0, 4.0, 5.0],       # Higher = more points but might include errors
        'merge_threshold': [0.001, 0.005, 0.01]          # For merge_triangulated_points
    }
}

# Function to generate all possible hyperparameter combinations
def generate_hyperparameter_combinations(grid):
    # Create lists of parameter combinations for each section
    section_combinations = {}
    for section, params in grid.items():
        section_keys = list(params.keys())
        section_values = list(itertools.product(*[params[key] for key in section_keys]))
        section_combinations[section] = [dict(zip(section_keys, values)) for values in section_values]
    
            # We're only using SIFT, so no special handling needed for ORB
    feature_combinations = []
    for combo in section_combinations['features']:
        feature_combinations.append(combo)
    section_combinations['features'] = feature_combinations
    
    # Generate all combinations across all sections
    all_combinations = []
    
    # Get all combinations of section combinations
    features_combos = section_combinations['features']
    matching_combos = section_combinations['matching']
    calibration_combos = section_combinations['calibration']
    sfm_combos = section_combinations['sfm']
    
    # Product of all section combinations
    for f_combo in features_combos:
        for m_combo in matching_combos:
            for c_combo in calibration_combos:
                for s_combo in sfm_combos:
                    # Create a copy of the base configuration
                    config = copy.deepcopy(base_config)
                    
                    # Update with the current combination
                    config['features'].update(f_combo)
                    config['matching'].update(m_combo)
                    config['calibration'].update(c_combo)
                    config['sfm'].update(s_combo)
                    
                    all_combinations.append(config)
    
    return all_combinations

# Main function to run reconstruction with a specific configuration
def run_reconstruction(config, dataset_path_black, dataset_path_original, output_dir, config_id):
    print(f"\n{'='*80}")
    print(f"Running configuration {config_id}")
    print(f"{'='*80}")
    
    # Create a specific output directory for this configuration
    config_output_dir = os.path.join(output_dir, f"config_{config_id}")
    os.makedirs(config_output_dir, exist_ok=True)
    
    # Save configuration to output directory
    with open(os.path.join(config_output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    # Create a parameter summary string for plot titles
    param_summary = (
        f"Features: {config['features']['max_features']}, "
        f"Ratio: {config['matching']['ratio_threshold']}, "
        f"MinMatches: {config['matching']['min_matches']}, "
        f"Focal: {config['calibration']['focal_length_factor']}, "
        f"TriangAngle: {config['sfm']['min_triangulation_angle_deg']}, "
        f"ReprojErr: {config['sfm']['reprojection_error_threshold']}"
    )
    
    # Create a shorter parameter summary for filenames
    param_filename = (
        f"F{config['features']['max_features']}_"
        f"R{config['matching']['ratio_threshold']}_"
        f"M{config['matching']['min_matches']}_"
        f"FL{config['calibration']['focal_length_factor']}_"
        f"TA{config['sfm']['min_triangulation_angle_deg']}_"
        f"RE{config['sfm']['reprojection_error_threshold']}"
    )
    
    # Start timing
    start_time = time.time()
    
    # Create a results dictionary to track metrics
    results = {
        'config_id': config_id,
        'parameters': param_summary,
        'start_time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'metrics': {}
    }
    
    try:
        # Step 1: Load and preprocess images
        print("Loading black background images...")
        black_images = load_image_sequence(dataset_path_black, pattern="viff.*.png")
        print(f"Loaded {len(black_images)} black background images.")
        
        print("Loading original background images...")
        original_images = load_image_sequence(dataset_path_original, pattern="viff.*.png")
        print(f"Loaded {len(original_images)} original background images.")
        
        # Make sure image lists are the same length and in the same order
        if len(black_images) != len(original_images):
            print("Warning: Different number of images in the two datasets")
            blackfilenames = [filename for _, filename in black_images]
            originalfilenames = [filename for _, filename in original_images]
            common_filenames = set(blackfilenames).intersection(set(originalfilenames))
            
            black_images = [(img, filename) for img, filename in black_images if filename in common_filenames]
            original_images = [(img, filename) for img, filename in original_images if filename in common_filenames]
            
            print(f"Using {len(black_images)} images that exist in both datasets")
        
        # Initialize points_3d to empty list in case triangulation fails
        points_3d = []
        point_observations = []
        
        # Resize images
        max_dim = config['preprocessing']['resize_max_dimension']
        black_images = resize_images(black_images, max_dimension=max_dim)
        original_images = resize_images(original_images, max_dimension=max_dim)
        
        # Enhance contrast if specified
        if config['preprocessing']['enhance_contrast']:
            black_images = enhance_contrast(black_images)
            original_images = enhance_contrast(original_images)
        
        # Step 2: Extract features
        print("\nExtracting features...")
        feature_method = config['features']['method']
        max_features = config['features']['max_features']
        
        # Extract features using the exact pattern from the original code
        feature_method = config['features']['method']
        max_features = config['features']['max_features']
        
        # Extract features exactly as in the original code
        if feature_method.lower() == 'sift':
            print(f"Using SIFT extraction with target of {max_features} features per image")
            features_dict = extract_features_from_image_set(black_images, method=feature_method, n_features=max_features)
        else:
            # For other methods, use standard extraction
            features_dict = extract_features_from_image_set(black_images, method=feature_method, n_features=max_features)
        
        # Count features
        total_features = sum(len(keypoints) for keypoints, _ in features_dict.values())
        avg_features = total_features / len(features_dict)
        print(f"Total features extracted: {total_features} (avg {avg_features:.0f} per image)")
        
        results['metrics']['total_features'] = total_features
        results['metrics']['avg_features_per_image'] = avg_features
        
        # Step 3: Create image pairs and match features
        filenames = sorted([filename for _, filename in black_images], 
                          key=lambda x: int(''.join(filter(str.isdigit, x))))
        
        # Number of images
        n = len(filenames)
        
        # Create image pairs for matching with a comprehensive circular strategy
        image_pairs = []
        
        # 1. Keep sequential pairs as your foundation
        for i in range(n-1):
            image_pairs.append((filenames[i], filenames[i+1]))
        image_pairs.append((filenames[-1], filenames[0]))  # Close the loop
        
        # Remove duplicates while preserving order
        seen = set()
        image_pairs = [x for x in image_pairs if not (x in seen or seen.add(x))]
        
        print(f"Created {len(image_pairs)} image pairs for matching")
        
        # Match features exactly as in the original code
        matches_dict = match_image_pairs(
            features_dict, 
            image_pairs, 
            ratio_threshold=config['matching']['ratio_threshold'],
            geometric_verify=config['matching']['geometric_verification'],
            min_matches=config['matching']['min_matches']
        )
        
        num_matched_pairs = len(matches_dict)
        print(f"Successfully matched {num_matched_pairs} image pairs.")
        results['metrics']['matched_image_pairs'] = num_matched_pairs
        
        # Step 4: Estimate camera intrinsics
        sample_img, _ = black_images[0]
        image_shape = sample_img.shape
        focal_length_factor = config['calibration']['focal_length_factor']
        focal_length = focal_length_factor * max(image_shape[0], image_shape[1])
        K = estimate_camera_matrix(image_shape, focal_length)
        
        # Step 5: Estimate camera poses
        camera_poses = estimate_poses_incremental(
            matches_dict, 
            K, 
            min_matches=config['matching']['min_matches']
        )
        
        num_camera_poses = len(camera_poses)
        print(f"Estimated poses for {num_camera_poses} cameras.")
        results['metrics']['camera_poses'] = num_camera_poses
        
        # Visualize and save camera poses plot
        if num_camera_poses > 0:
            # Save camera poses as static plot
            plt.figure(figsize=(10, 8))
            
            # Extract camera centers for plotting
            camera_centers = {}
            for name, (R, t) in camera_poses.items():
                center = -R.T @ t
                camera_centers[name] = center
            
            # Plot camera centers
            centers = np.array(list(camera_centers.values()))
            plt.scatter(centers[:, 0], centers[:, 1], c='r', marker='o', s=50)
            
            plt.title(f"Camera Poses (Top View)\n{param_summary}")
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.grid(True)
            
            camera_plot_path = os.path.join(config_output_dir, f"camera_poses_{param_filename}.png")
            plt.savefig(camera_plot_path, dpi=300)
            plt.close()
            
            results['metrics']['camera_plot_path'] = camera_plot_path
            
            # Also save the 3D point cloud if available
            if points_3d and len(points_3d) > 0:
                # Create a static plot of the point cloud (top view)
                plt.figure(figsize=(12, 10))
                points_array = np.array(points_3d)
                plt.scatter(points_array[:, 0], points_array[:, 1], s=1, alpha=0.5, c='blue')
                plt.scatter(centers[:, 0], centers[:, 1], c='r', marker='o', s=50)
                plt.title(f"Sparse 3D Reconstruction: {len(points_array)} points (Top View)\n{param_summary}")
                plt.xlabel('X')
                plt.ylabel('Y')
                plt.grid(True)
                
                point_cloud_path = os.path.join(config_output_dir, f"sparse_cloud_{param_filename}.png")
                plt.savefig(point_cloud_path, dpi=300)
                plt.close()
                
                results['metrics']['point_cloud_plot_path'] = point_cloud_path
        
        # Step 6: Triangulate 3D points
        points_3d, point_observations = triangulate_all_points(
            camera_poses, 
            matches_dict, 
            K,
            min_angle_deg=config['sfm']['min_triangulation_angle_deg'],
            max_reproj_error=config['sfm']['max_reprojection_error']
        )
        
        num_triangulated_points = len(points_3d)
        print(f"Triangulated {num_triangulated_points} 3D points.")
        results['metrics']['triangulated_points'] = num_triangulated_points
        
        # Merge close points
        merged_points, merged_observations = merge_triangulated_points(
            points_3d, 
            point_observations, 
            threshold=config['sfm']['merge_threshold']
        )
        
        num_merged_points = len(merged_points)
        print(f"After merging: {num_merged_points} 3D points.")
        results['metrics']['merged_points'] = num_merged_points
        
        # Step 7: Bundle adjustment
        if config['sfm']['refine_poses'] and num_merged_points > 0:
            print("\nRunning bundle adjustment...")
            refined_poses, refined_points, _ = run_global_ba(
                camera_poses, 
                matches_dict, 
                K, 
                iterations=20
            )
            points_3d = refined_points
            
            num_refined_points = len(refined_points)
            print(f"Bundle adjustment complete with {num_refined_points} refined points.")
            results['metrics']['refined_points'] = num_refined_points
        
        # Step 8: Visualize and save sparse point cloud
        if len(points_3d) > 0:
            points_array = np.array(points_3d)
            
            # Assign random colors for visualization
            np.random.seed(42)  # For reproducibility
            colors = np.random.rand(len(points_array), 3)
            
            # Visualize sparse point cloud
            if len(points_3d) > 0:
                points_array = np.array(points_3d)
                
                # Assign random colors for visualization
                np.random.seed(42)  # For reproducibility
                colors = np.random.rand(len(points_array), 3)
                
                # Save sparse point cloud as PLY
                sparse_cloud_file = os.path.join(config_output_dir, f"sparse_cloud_{param_filename}.ply")
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points_array)
                pcd.colors = o3d.utility.Vector3dVector(colors)
                o3d.io.write_point_cloud(sparse_cloud_file, pcd)
                print(f"Saved sparse point cloud to {sparse_cloud_file}")
                
                results['metrics']['point_cloud_file'] = sparse_cloud_file
        
        # Record end time and duration
        end_time = time.time()
        duration = end_time - start_time
        results['duration_seconds'] = duration
        results['end_time'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"\nConfiguration {config_id} completed in {duration:.2f} seconds")
        print(f"Points triangulated: {num_triangulated_points}")
        
        # Calculate a simple quality score (could be refined based on your specific needs)
        quality_score = num_merged_points * (num_camera_poses / len(black_images))
        results['quality_score'] = quality_score
        print(f"Quality score: {quality_score:.2f}")
        
        return results
        
    except Exception as e:
        error_msg = f"Error in configuration {config_id}: {str(e)}"
        print(error_msg)
        results['error'] = error_msg
        return results

# Main execution function
def run_hyperparameter_sweep():
    # Define paths to datasets
    dataset_path_black = '../data/dinosaur_cropped_black/'  # Black background
    dataset_path_original = '../data/dinosaur_cropped/'     # Original background
    
    # Create main output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"../data/results/hyperparameter_sweep_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate all hyperparameter combinations
    all_configs = generate_hyperparameter_combinations(hyperparameter_grid)
    print(f"Generated {len(all_configs)} configurations to test")
    
    # Save all configurations to a single file for reference
    with open(os.path.join(output_dir, 'all_configurations.json'), 'w') as f:
        json.dump(all_configs, f, indent=4)
    
    # Create a results list to track performance of each configuration
    all_results = []
    
    # Track the best configuration by number of triangulated points
    best_config = None
    best_score = 0
    
    # Run each configuration
    for i, config in enumerate(tqdm(all_configs, desc="Testing configurations")):
        print(f"\nConfiguration {i+1}/{len(all_configs)}")
        
        # Run reconstruction with this configuration
        results = run_reconstruction(
            config, 
            dataset_path_black, 
            dataset_path_original, 
            output_dir, 
            i+1
        )
        
        # Save the results
        all_results.append(results)
        
        # Update the best configuration if applicable
        if 'quality_score' in results and results['quality_score'] > best_score:
            best_score = results['quality_score']
            best_config = i+1
            print(f"New best configuration: {i+1} with score {best_score:.2f}")
        
        # Save the current results to disk after each configuration
        with open(os.path.join(output_dir, 'results_summary.json'), 'w') as f:
            json.dump({
                'total_configs': len(all_configs),
                'configs_tested': i+1,
                'best_config': best_config,
                'best_score': best_score,
                'results': all_results
            }, f, indent=4)
        
        # Create a simple results table
        results_df = {}
        for res in all_results:
            config_id = res.get('config_id', 'unknown')
            results_df[config_id] = {
                'triangulated_points': res.get('metrics', {}).get('triangulated_points', 0),
                'merged_points': res.get('metrics', {}).get('merged_points', 0),
                'camera_poses': res.get('metrics', {}).get('camera_poses', 0),
                'quality_score': res.get('quality_score', 0),
                'parameters': res.get('parameters', '')
            }
        
        # Convert to DataFrame and save as CSV
        import pandas as pd
        pd_results = pd.DataFrame.from_dict(results_df, orient='index')
        pd_results.sort_values('quality_score', ascending=False, inplace=True)
        pd_results.to_csv(os.path.join(output_dir, 'results_table.csv'))
        
        # Print the current top 5 configurations
        print("\nCurrent Top 5 Configurations:")
        print(pd_results.head(5)[['triangulated_points', 'merged_points', 'camera_poses', 'quality_score']])
    
    print("\n\nHyperparameter sweep completed!")
    if best_config is not None:
        print(f"Best configuration: {best_config} with quality score {best_score:.2f}")
    
    return all_results

# Run the hyperparameter sweep
results = run_hyperparameter_sweep()