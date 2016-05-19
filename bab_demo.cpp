#include "bab_gui.cpp"

int main(int argc, char * argv[])
{

	// Get parameters from command line.
	std::vector<float> solutions;
	std::vector<float> intervals;
    solutions.push_back(0.0f);
    solutions.push_back(5.0f);
    solutions.push_back(0.0f);
    solutions.push_back(5.0f);
    solutions.push_back(0.0f);
    solutions.push_back(5.0f);
    
    intervals.push_back(-5.0f);
    intervals.push_back(0.0f);
    intervals.push_back(-5.0f);
    intervals.push_back(0.0f);
    intervals.push_back(-5.0f);
    intervals.push_back(0.0f);

	int dimensions = 3;

	Level_Set_GUI &LS_GUI = Level_Set_GUI::getInstance();
	LS_GUI.dimensions = dimensions;
	LS_GUI.update_candidates(intervals);
	LS_GUI.update_solutions(solutions);

    LS_GUI.setup(argc, argv);
    
    LS_GUI.mainLoop();
    

	return 0;
}
