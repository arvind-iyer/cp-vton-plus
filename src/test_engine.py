from vton_service import InferenceEngine as VTON


if __name__ == "__main__":
    person_image_path = "../data/person.jpg"
    person_pose_path = "../data/person_pose.json"
    gmm_model_path = "../checkpoints/GMM/gmm_final.pth"
    tom_model_path = "../checkpoints/TOM/tom_final.pth"
    engine = VTON(gmm_model_path, tom_model_path)
    engine.load()
    result = engine.infer(person_image_path, "01", open(person_pose_path, 'r'))
    # result.save('../output/test/tryon.jpg')
