import os

def greet_user():
    print("Hello! I'm your cognitive neuroscience research assistant.")
    print("I'm here to help you with brain imaging analysis, coding, and high-performance computing.")
    print("Let's get started on your project.")

def get_project_info():
    project_description = input("First, could you briefly describe your current project? ")
    imaging_modality = input("What type of brain imaging data are you working with (e.g., fMRI, EEG, DTI)? ")
    analysis_type = input("What specific analysis are you trying to perform? ")
    return project_description, imaging_modality, analysis_type

def get_user_experience():
    programming_experience = input("What is your level of programming experience (e.g., Python, MATLAB, R)? ")
    hpc_familiarity = input("How familiar are you with HPC environments or Git? ")
    return programming_experience, hpc_familiarity

def provide_guidance(imaging_modality, analysis_type, programming_experience, hpc_familiarity):
    print("\nThank you for sharing that information. Based on your project, here’s how I can assist you:")
    
    # Placeholder for analysis explanation
    print(f"\n1. Explaining the {analysis_type} analysis for your {imaging_modality} data:")
    print("   - I can break down the complex concepts into understandable components.")
    print("   - I can also provide references to relevant academic papers.")

    # Placeholder for code development guidance
    print("\n2. Code Development:")
    print("   - I will guide you in developing efficient and reproducible code following best practices.")
    
    # Placeholder for HPC/SLURM guidance
    if "hpc" in hpc_familiarity.lower() or "slurm" in hpc_familiarity.lower():
        print("\n3. HPC and SLURM:")
        print("   - I can provide instructions for executing your code on Stanford's HPC cluster using SLURM.")
        print("   - We can also discuss efficient resource management.")

    # Placeholder for Git repository management
    if "git" in hpc_familiarity.lower():
        print("\n4. Git Repository Management:")
        print("   - I can help you set up and use a Git repository for your project.")

def main():
    greet_user()
    project_description, imaging_modality, analysis_type = get_project_info()
    programming_experience, hpc_familiarity = get_user_experience()
    provide_guidance(imaging_modality, analysis_type, programming_experience, hpc_familiarity)

if __name__ == "__main__":
    main()

