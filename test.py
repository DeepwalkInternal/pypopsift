import pypopsift
import numpy as np
import sys

def run_comprehensive_test():
    """
    A comprehensive test script for the pypopsift library bindings.
    """
    print("--- Running Comprehensive PopSift Test ---")

    try:
        config = pypopsift.Config()
        config.octaves = 5
        config.levels = 3
        config.setThreshold(0.01)
        config.setEdgeLimit(10.0)
        config.set_sift_mode(pypopsift.SiftMode.PopSift)
        print("✅ Config object created and configured successfully.")
    except Exception as e:
        print(f"❌ Error during configuration: {e}")
        sys.exit(1)

    try:
        sift = pypopsift.PopSift(config, pypopsift.ProcessingMode.MatchingMode)
        print("✅ PopSift object initialized successfully.")
    except Exception as e:
        print(f"❌ Error initializing PopSift: {e}")
        sys.exit(1)

    width, height = 512, 512
    image1_data = np.random.randint(0, 256, (height, width), dtype=np.uint8)
    image2_data = np.roll(image1_data, shift=(0, 50), axis=(1, 0))
    print(f"✅ Created two {width}x{height} test images.")

    try:
        print("Enqueueing image 1...")
        # --- Corrected this line ---
        job1 = sift.enqueue(image1_data) 
        print("Enqueueing image 2...")
        # --- And this line ---
        job2 = sift.enqueue(image2_data) 
        print("✅ Both images enqueued successfully.")
    except Exception as e:
        print(f"❌ Error enqueuing images: {e}")
        sift.uninit()
        sys.exit(1)

    try:
        print("Waiting for jobs to complete...")
        features1_dev = job1.getDev()
        features2_dev = job2.getDev()
        print("✅ Features retrieved to the device successfully.")
        
        num_features1 = features1_dev.getFeatureCount()
        num_features2 = features2_dev.getFeatureCount()
        print(f"--- Results for Image 1 (on device) ---")
        print(f"    - Found {num_features1} feature points.")
        print(f"--- Results for Image 2 (on device) ---")
        print(f"    - Found {num_features2} feature points.")

    except Exception as e:
        print(f"❌ Error retrieving or inspecting features: {e}")
    
    try:
        print("\n--- Performing GPU Feature Matching ---")
        if features1_dev.getFeatureCount() > 0 and features2_dev.getFeatureCount() > 0:
            features1_dev.match(features2_dev)
            print("✅ GPU feature matching completed (check console for match output).")
        else:
            print("⚠️ Skipping matching because one of the images has no features.")
            
    except Exception as e:
        print(f"❌ Error during feature matching: {e}")

    finally:
        sift.uninit()
        print("\n✅ PopSift resources released.")

    print("\n--- Comprehensive Test Complete ---")

if __name__ == "__main__":
    run_comprehensive_test()
