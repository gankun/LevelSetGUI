#include "bab_gui.cpp"
#include <unistd.h>


std::vector<float> solutions;
std::vector<float> intervals;
Level_Set_GUI  LS_GUI_G;
int main(int argc, char * argv[])
{

	// Get parameters from command line.
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

    intervals.push_back(-10.0f);
    intervals.push_back(10.0f);
    intervals.push_back(-10.0f);
    intervals.push_back(10.0f);
    intervals.push_back(-10.0f);
    intervals.push_back(10.0f);
    
    intervals.push_back(-10.0f);
    intervals.push_back(-5.0f);
    intervals.push_back(-10.0f);
    intervals.push_back(-5.0f);
    intervals.push_back(-10.0f);
    intervals.push_back(-5.0f);

    int dimensions = 3;

	Level_Set_GUI &LS_GUI = Level_Set_GUI::getInstance();
	LS_GUI.dimensions = dimensions;
	LS_GUI.update_candidates(intervals);
	LS_GUI.update_solutions(solutions);

    LS_GUI.setup(argc, argv);

//    LS_GUI_G = LS_GUI;


    LS_GUI.mainLoop();
}      


void foo ()
{

//    LS_GUI.display();

    usleep(3000 * 1000);
   
    intervals.push_back(0.0f);
    intervals.push_back(2.5f);
    intervals.push_back(0.0f);
    intervals.push_back(2.5f);
    intervals.push_back(.0f);
    intervals.push_back(2.5f);
    

//    LS_GUI.update_candidates(intervals);




  //  LS_GUI->display();

    usleep(3000 * 1000);

//    LS_GUI.mainLoop();
//	return 0;
}
