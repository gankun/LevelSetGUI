

#include <GL/glew.h>
#include <GL/glut.h>
#include <Eigen/Eigen>

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>

#include <string>
#include <list>
#include <vector>

using namespace Eigen;
using namespace std;

// Class for GUI for interval analysis.

/* Gets the rotation matrix for a given rotation axis (x, y, z) and radian */
MatrixXd get_rotate_mat(float x, float y, float z, float rad)
{

    float norm = x * x + y * y + z * z;
    norm = sqrt(norm);
    x = x / norm;
    y = y / norm;
    z = z / norm;
    //float rad = (angle * 3.14159265 / 180.0);
    //angle = angle * 180 / 3.14159265;
    float cos_rad = cos(rad);
    float sin_rad = sin(rad);

    MatrixXd rot;
    rot = MatrixXd::Identity(4,4);
    rot(0,0) = x * x + (1 - (x * x)) * cos_rad;
    rot(0,1) = x * y * (1 - cos_rad) - z * sin_rad;
    rot(0,2) = x * z * (1 - cos_rad) + y * sin_rad;
    rot(1,1) = y * y + (1 - (y * y)) * cos_rad;
    rot(1,0) = x * y * (1 - cos_rad) + z * sin_rad;
    rot(1,2) = y * z * (1 - cos_rad) - x * sin_rad;
    rot(2,0) = x * z * (1 - cos_rad) - y * sin_rad;
    rot(2,1) = z * y * (1 - cos_rad) + x * sin_rad;
    rot(2,2) = z * z + (1 - (z * z)) * cos_rad;
    return rot;
}


class Level_Set_GUI
{
	// Class Methods and Variables
	public:
		Level_Set_GUI(void);
		Level_Set_GUI(int n_dimensions); // Constructor
		~Level_Set_GUI(); // Destructor
		void setup(int argc, char * argv[]);
        void mainLoop();
		
        void update_candidates(vector<float> new_intervals);
        void update_solutions(vector<float> new_intervals);
        
		static Level_Set_GUI& getInstance() // Singleton is accessed via getInstance()
		{
			static Level_Set_GUI instance; // lazy singleton, instantiated on first use
			return instance;
		}

		int dimensions;




	private:
        static Level_Set_GUI* current_instance;
    
        int xres = 500;
        int yres = 500;
        
        float * box_color(float * color);
        
        static void display();
        static void key_pressed(unsigned char key, int x, int y);
        static void mouse_pressed(int button, int state, int x, int y);
        static void mouse_moved(int x, int y);
        static void reshape(int width, int height);
        
        void displayImpl();		
        void key_pressedImpl(unsigned char key, int x, int y);
        void mouse_pressedImpl(int button, int state, int x, int y);
        void mouse_movedImpl(int x, int y);
        void reshapeImpl(int width, int height);
        
        void draw_boxes(std::vector<float>, int, float *, bool);
        
        Vector3d get_arc_vec(int x, int y);

        		// Intervals Vector
		std::vector<float> candidate_vector;
		// Intervals Solutions
		std::vector<float> solution_vector;
        
		/* Flag deciding whether to draw interval/solution boxes */
		bool drawSolution = true;
		bool drawInterval = true;

		bool fillSolution = false;

		int iter;
		int * inter_array;

		float lineWidth = .4f;

		/* Acrball rotation Matrix */
		//Matrix4f last_rotation = Matrix4f::Identity();
		//Matrix4f current_rotation = Matrix4f::Identity();

		/* Camera Settings */
		float cam_orientation_angle = 0;
        float cam_orientation_axis[3];
		float x_view_angle = 20, y_view_angle = 20;
		float cam_position[3];


		/* Fulstrum */
		float near_param = 1, far_param = 100,
			left_param = -0.5, right_param = 0.5,
			top_param = 0.5, bottom_param = -0.5;

		const float step_size = 0.2;
		const float x_view_step = 90.0, y_view_step = 90.0;


		const int margin = 20;
		const int ticksize = 10;
		float offset_x = 0.0;
		float scale_x = 1.0;

		double mouse_x, mouse_y;
		double last_mx = 0;
		double last_my = 0;
        float mouse_scale_x, mouse_scale_y;
		MatrixXd last_rotation;
		MatrixXd current_rotation;
        Vector3d axis_in_cam;
        
		bool mousePressed = false;


};

Level_Set_GUI::Level_Set_GUI(void){

	dimensions = 2;

	// Initialize arrays
	cam_orientation_axis[0] = .75;
	cam_orientation_axis[1] = .75;
	cam_orientation_axis[2] = 1;
	cam_position[0] = 0;
	cam_position[1] = -5;
	cam_position[2] = 10;

	current_rotation = MatrixXd::Identity(4, 4);
}

Level_Set_GUI::~Level_Set_GUI(){

	
}

void Level_Set_GUI::update_candidates(vector<float> new_intervals)
{
    candidate_vector = new_intervals;
}

void Level_Set_GUI::update_solutions(vector<float> new_intervals)
{
    solution_vector = new_intervals;
}

void Level_Set_GUI::mainLoop()
{
    glutMainLoop();
}

void Level_Set_GUI::setup(int argc, char * argv[]) {

    glutInit(&argc, argv);

	glEnable(GL_DEPTH_TEST);
    
    // Tell OpenGL that we need a double, RGB pixel, and a depth buffer.
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);

    glutInitWindowSize(xres, yres);
    
    // Set the program window in the top-left corner 0,0
    glutInitWindowPosition(0, 0);
     // Name the program window "Renderer".
    glutCreateWindow("Renderer");
    
    glShadeModel(GL_SMOOTH);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glEnable(GL_DEPTH_TEST);
    
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_NORMAL_ARRAY);
    
    glMatrixMode(GL_PROJECTION);
    
    glLoadIdentity();
    
    glFrustum(left_param, right_param,
		bottom_param, top_param,
		near_param, far_param);
    
    glMatrixMode(GL_MODELVIEW);
   
    
      /* Specify to OpenGL our display function.
     */
    glutDisplayFunc(&Level_Set_GUI::display);

    /* Specify to OpenGL our reshape function.
     */
    glutReshapeFunc(&Level_Set_GUI::reshape);

    /* Specify to OpenGL our function for handling mouse presses.
     */
    glutMouseFunc(&Level_Set_GUI::mouse_pressed);
    /* Specify to OpenGL our function for handling mouse movement.
     */
    glutMotionFunc(&Level_Set_GUI::mouse_moved);
    /* Specify to OpenGL our function for handling key presses.
     */
    glutKeyboardFunc(&Level_Set_GUI::key_pressed);
    
}

void Level_Set_GUI::reshape(int width, int height)
{
    getInstance().reshapeImpl(width, height);
    //current_instance->reshape(width, height);
}

void Level_Set_GUI::reshapeImpl(int width, int height)
{
    /* The following two lines of code prevent the width and height of the
     * window from ever becoming 0 to prevent divide by 0 errors later.
     * Typically, we let 1x1 square pixel be the smallest size for the window.
     */
    height = (height == 0) ? 1 : height;
    width = (width == 0) ? 1 : width;
    
    /* The 'glViewport' function tells OpenGL to determine how to convert from
     * NDC to screen coordinates given the dimensions of the window. The
     * parameters for 'glViewport' are (in the following order):
     *
     * - int x: x-coordinate of the lower-left corner of the window in pixels
     * - int y: y-coordinate of the lower-left corner of the window in pixels
     * - int width: width of the window
     * - int height: height of the window
     *
     * We typically just let the lower-left corner be (0,0).
     *
     * After 'glViewport' is called, OpenGL will automatically know how to
     * convert all our points from NDC to screen coordinates when it tries
     * to render them.
     */
    glViewport(0, 0, width, height);
    
    /* The following two lines are specific to updating our mouse interface
     * parameters. Details will be given in the 'mouse_moved' function.
     */
    mouse_scale_x = (float) (right_param - left_param) / (float) width;
    mouse_scale_y = (float) (top_param - bottom_param) / (float) height;
    
    /* The following line tells OpenGL that our program window needs to
     * be re-displayed, meaning everything that was being displayed on
     * the window before it got resized needs to be re-rendered.
     */
    glutPostRedisplay();
}

void Level_Set_GUI::key_pressed(unsigned char key, int x, int y)
{
    getInstance().key_pressedImpl(key, x, y);
    //current_instance->key_pressedImpl(key, x, y);
}

void Level_Set_GUI::key_pressedImpl(unsigned char key, int x, int y)
{
    switch (key)
    {
        case 27: // ESCAPE Key
            exit(0);
            break;
      
        case 'q':
            cam_orientation_angle += .2;
            break;
            
        case 'e':
            cam_orientation_angle -= .2;
            break;
            
        case 'w':
            cam_position[1] += .3F;
            break;
            
        case 's':
            cam_position[1] -= .3F;
            break;
            
        case 'a':
            cam_position[0] -= .3F;
            break;
            
        case 'd':
            cam_position[0] += .3F;
            break;
            
        case 'z':
            cam_position[2] -= .3F;
            break;
            
        case 'c':
            cam_position[2] += .3F;
            break;
            
        case 'f':
            fillSolution = !fillSolution;
            break;
    }
}


float deg2rad(float angle)
{
    return angle * M_PI / 180.0;
}

/* 'rad2deg' function:
 * 
 * Converts given angle in radians to degrees.
 */
float rad2deg(float rad)
{
    return rad * 180.0 / M_PI;
}

Vector3d Level_Set_GUI::get_arc_vec(int x, int y)
{
    Vector3d P;
    P << 1.0 * x / xres / 2 * 2 - 1.0,
         1.0 * y / yres / 2 * 2 - 1.0,
         0.0;
    
    P[1] *= -1;
    
    float OP_squared = P[0] * P[0] + P[1] * P[1];
    if (OP_squared <= 1 * 1)
        P[2] = sqrt(1*1 - OP_squared);
    else
    {
        P[0] = P[0] / sqrt(OP_squared);
        P[1] = P[1] / sqrt(OP_squared);
    }
    return P;
    
}
void Level_Set_GUI::display()
{
    getInstance().displayImpl();
    //current_instance->displayImpl();
}
void Level_Set_GUI::displayImpl()
{
    
    glutPostRedisplay();
    
    // clear the color and depth buffer
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();
    
    // Rotate the camera and translate to its position.
    glRotatef((-cam_orientation_angle * 180.0 / M_PI), cam_orientation_axis[0],
        cam_orientation_axis[1], cam_orientation_axis[2]);
    
    glTranslatef(cam_position[0],cam_position[1], cam_position[2]);
  /*  
    // Change the camera position and angle based on mouse movement.
    if (mouse_x != last_mx || mouse_y != last_my) 
    {
        Vector3d va, vb;
        // compute the arcball angle and axis
        va = get_arc_vec(last_mx, last_my); 
        vb = get_arc_vec(mouse_x, mouse_y); 
        float angle = acos(min(1.0, va.dot(vb)));
        axis_in_cam = va.cross(vb);
        
        current_rotation = (get_rotate_mat(axis_in_cam(0), 
            axis_in_cam(1), axis_in_cam(2), angle));
            
        // keep track of the rotations in last_rotation
        last_rotation = current_rotation * last_rotation;

        // save the last location
        last_mx = mouse_x;
        last_my = mouse_y;
    }
    
    glMultMatrixd(&last_rotation(0,0));
*/
	glViewport(
		margin + ticksize,
		margin + ticksize,
		xres - margin * 2 - ticksize,
		yres - margin * 2 - ticksize
		);

	glScissor(
		margin + ticksize,
		margin + ticksize,
		xres - margin * 2 - ticksize,
		yres - margin * 2 - ticksize
		);

	glEnable(GL_SCISSOR_TEST);
	
	// White background
	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);
	
    // test cube
    glBegin(GL_POLYGON);
    glColor3f(0.0f, 0.4f, 0.8f);

    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, 6.0f, 0.0f);
    glVertex3f(6.0f, 6.0f, 0.0f);
    glVertex3f(6.0f, 0.0f, 0.0f);

    glEnd();
    
	if (drawInterval)
	{
		float colorB[3] = { 0.1f, 0.1f, 0.7f };
		draw_boxes(candidate_vector, dimensions, colorB, false);
	}

	if (drawSolution)
	{
		float colorA[3] = { 0.7f, 0.1f, 0.1f };
		draw_boxes(solution_vector, dimensions, colorA, fillSolution);
	}


	/* Swap front and back buffers */
	glutSwapBuffers();

}

float * Level_Set_GUI::box_color(float * color)
{
    return color;
}

void Level_Set_GUI::draw_boxes(std::vector<float> intervals, int n, float * color, bool fillIn)
{
    float * tempColor;
	/* Go through each set of intervals in the vector and draw them. */
	for (int i = 0; i < intervals.size(); i += n * 2)
	{
		if (fillIn)
		{
			if (n == 3)
			{
				Vector3f A, B, C, Norm;
				float R, G, Bl;
				//Vector3f A, B, C, Norm;
				glLineWidth((GLfloat)lineWidth);

				// Draw the 3d Cube.
				glBegin(GL_POLYGON);

				A = { intervals[i], intervals[i + 2], intervals[i + 4] };
				B = {intervals[i], intervals[i + 3], intervals[i + 4]};
				C = { intervals[i + 1], intervals[i + 2], intervals[i + 4] };

                tempColor = box_color(color);
				glColor3f(tempColor[0], tempColor[1], tempColor[2]);

				// Draw front square.
				glVertex3f(intervals[i], intervals[i + 2], intervals[i + 4]);
				glVertex3f(intervals[i], intervals[i + 3], intervals[i + 4]);
				glVertex3f(intervals[i + 1], intervals[i + 2], intervals[i + 4]);
				glVertex3f(intervals[i + 1], intervals[i + 3], intervals[i + 4]);
				glEnd();
				glBegin(GL_POLYGON);


				A = { intervals[i], intervals[i + 2], intervals[i + 4] };
				B = { intervals[i + 1], intervals[i + 2], intervals[i + 4] };
				C = { intervals[i], intervals[i + 3], intervals[i + 4] };

				tempColor = box_color(color);
				glColor3f(tempColor[0], tempColor[1], tempColor[2]);

				glVertex3f(intervals[i], intervals[i + 2], intervals[i + 4]);
				glVertex3f(intervals[i + 1], intervals[i + 2], intervals[i + 4]);
				glVertex3f(intervals[i], intervals[i + 3], intervals[i + 4]);
				glVertex3f(intervals[i + 1], intervals[i + 3], intervals[i + 4]);
				glEnd();
				glBegin(GL_POLYGON);


				A = { intervals[i], intervals[i + 2], intervals[i + 5] };
				B = { intervals[i], intervals[i + 3], intervals[i + 5] };
				C = { intervals[i + 1], intervals[i + 2], intervals[i + 5] };

				tempColor = box_color(color);
				glColor3f(tempColor[0], tempColor[1], tempColor[2]);;


				// Draw back square.
				glVertex3f(intervals[i], intervals[i + 2], intervals[i + 5]);
				glVertex3f(intervals[i], intervals[i + 3], intervals[i + 5]);
				glVertex3f(intervals[i + 1], intervals[i + 2], intervals[i + 5]);
				glVertex3f(intervals[i + 1], intervals[i + 3], intervals[i + 5]);
				glEnd();
				glBegin(GL_POLYGON);

				A = { intervals[i], intervals[i + 2], intervals[i + 5] };
				B = { intervals[i + 1], intervals[i + 2], intervals[i + 5] };
				C = { intervals[i], intervals[i + 3], intervals[i + 5] };

				tempColor = box_color(color);
				glColor3f(tempColor[0], tempColor[1], tempColor[2]);

				glVertex3f(intervals[i], intervals[i + 2], intervals[i + 5]);
				glVertex3f(intervals[i + 1], intervals[i + 2], intervals[i + 5]);
				glVertex3f(intervals[i], intervals[i + 3], intervals[i + 5]);
				glVertex3f(intervals[i + 1], intervals[i + 3], intervals[i + 5]);
				glEnd();
				glBegin(GL_POLYGON);
				

				A = { intervals[i], intervals[i + 2], intervals[i + 4] };
				B = { intervals[i], intervals[i + 2], intervals[i + 5] };
				C = { intervals[i + 1], intervals[i + 3], intervals[i + 4] };

				tempColor = box_color(color);
				glColor3f(tempColor[0], tempColor[1], tempColor[2]);

				// Draw lines between squares.
				glVertex3f(intervals[i], intervals[i + 2], intervals[i + 4]);
				glVertex3f(intervals[i], intervals[i + 2], intervals[i + 5]);
				glVertex3f(intervals[i + 1], intervals[i + 3], intervals[i + 4]);
				glVertex3f(intervals[i + 1], intervals[i + 3], intervals[i + 5]);
				glEnd();
				glBegin(GL_POLYGON);

				tempColor = box_color(color);
				glColor3f(tempColor[0], tempColor[1], tempColor[2]);
                
				glVertex3f(intervals[i + 1], intervals[i + 2], intervals[i + 4]);
				glVertex3f(intervals[i + 1], intervals[i + 2], intervals[i + 5]);
				glVertex3f(intervals[i], intervals[i + 3], intervals[i + 4]);
				glVertex3f(intervals[i], intervals[i + 3], intervals[i + 5]);

				glEnd();
				
			}
		}
		else
		{
			if (n == 2)	{
				glLineWidth((GLfloat)lineWidth);
				glBegin(GL_LINES);
				glColor3f(color[0], color[1], color[2]);
				glVertex3f(intervals[i], intervals[i + 2], 0.0f);
				glVertex3f(intervals[i], intervals[i + 3], 0.0f);

				glVertex3f(intervals[i + 1], intervals[i + 2], 0.0f);
				glVertex3f(intervals[i + 1], intervals[i + 3], 0.0f);

				glVertex3f(intervals[i], intervals[i + 2], 0.0f);
				glVertex3f(intervals[i + 1], intervals[i + 2], 0.0f);

				glVertex3f(intervals[i], intervals[i + 3], 0.0f);
				glVertex3f(intervals[i + 1], intervals[i + 3], 0.0f);
				glEnd();
			}
			if (n == 3){
				glLineWidth((GLfloat)lineWidth);
				// Draw the 3d Cube.
				glBegin(GL_LINES);
				glColor3f(color[0], color[1], color[2]);

				// Draw front square.
				glVertex3f(intervals[i], intervals[i + 2], intervals[i + 4]);
				glVertex3f(intervals[i], intervals[i + 3], intervals[i + 4]);
				glVertex3f(intervals[i + 1], intervals[i + 2], intervals[i + 4]);
				glVertex3f(intervals[i + 1], intervals[i + 3], intervals[i + 4]);
				glVertex3f(intervals[i], intervals[i + 2], intervals[i + 4]);
				glVertex3f(intervals[i + 1], intervals[i + 2], intervals[i + 4]);
				glVertex3f(intervals[i], intervals[i + 3], intervals[i + 4]);
				glVertex3f(intervals[i + 1], intervals[i + 3], intervals[i + 4]);
				// Draw back square.
				glVertex3f(intervals[i], intervals[i + 2], intervals[i + 5]);
				glVertex3f(intervals[i], intervals[i + 3], intervals[i + 5]);
				glVertex3f(intervals[i + 1], intervals[i + 2], intervals[i + 5]);
				glVertex3f(intervals[i + 1], intervals[i + 3], intervals[i + 5]);
				glVertex3f(intervals[i], intervals[i + 2], intervals[i + 5]);
				glVertex3f(intervals[i + 1], intervals[i + 2], intervals[i + 5]);
				glVertex3f(intervals[i], intervals[i + 3], intervals[i + 5]);
				glVertex3f(intervals[i + 1], intervals[i + 3], intervals[i + 5]);
				// Draw lines between squares.
				glVertex3f(intervals[i], intervals[i + 2], intervals[i + 4]);
				glVertex3f(intervals[i], intervals[i + 2], intervals[i + 5]);
				glVertex3f(intervals[i + 1], intervals[i + 3], intervals[i + 4]);
				glVertex3f(intervals[i + 1], intervals[i + 3], intervals[i + 5]);
				glVertex3f(intervals[i + 1], intervals[i + 2], intervals[i + 4]);
				glVertex3f(intervals[i + 1], intervals[i + 2], intervals[i + 5]);
				glVertex3f(intervals[i], intervals[i + 3], intervals[i + 4]);
				glVertex3f(intervals[i], intervals[i + 3], intervals[i + 5]);

				glEnd();
			}
		}
	}
}
void Level_Set_GUI::mouse_pressed(int button, int state, int x, int y)
{
    getInstance().mouse_pressedImpl(button, state, x, y);
    //current_instance->mouse_pressedImpl(button, state, x, y);
}
void Level_Set_GUI::mouse_pressedImpl(int button, int state, int x, int y)
{
    /* If the left-mouse button was clicked down, then...
     */
    if(button == GLUT_LEFT_BUTTON && state == GLUT_DOWN)
    {
        /* Store the mouse position in our global variables.
         */
        mouse_x = x;
        mouse_y = y;
        last_mx = x;
        last_my = y;
        /* Since the mouse is being pressed down, we set our 'is_pressed"
         * boolean indicator to true.
         */
        mousePressed = true;
    }
    /* If the left-mouse button was released up, then...
     */
    else if(button == GLUT_LEFT_BUTTON && state == GLUT_UP)
    {
        /* Mouse is no longer being pressed, so set our indicator to false.
         */
        mousePressed = false;
    }
}

void Level_Set_GUI::mouse_moved(int x, int y)
{
    getInstance().mouse_movedImpl(x, y);
    //current_instance->mouse_movedImpl(x, y);
}
void Level_Set_GUI::mouse_movedImpl(int x, int y)
{
    /* If the left-mouse button is being clicked down...
     */
    if(mousePressed)
    {
        /* You see in the 'mouse_pressed' function that when the left-mouse button
         * is first clicked down, we store the screen coordinates of where the
         * mouse was pressed down in 'mouse_x' and 'mouse_y'. When we move the
         * mouse, its screen coordinates change and are captured by the 'x' and
         * 'y' parameters to the 'mouse_moved' function. We want to compute a change
         * in our camera angle based on the distance that the mouse traveled.
         *
         * Let's start with the horizontal angle change. We first need to convert
         * the dx traveled in screen coordinates to a dx traveled in camera space.
         * The conversion is done using our 'mouse_scale_x' variable, which we
         * set in our 'reshape' function. We then multiply by our 'x_view_step'
         * variable, which is an arbitrary value that determines how "fast" we
         * want the camera angle to change. Higher values for 'x_view_step' cause
         * the camera to move more when we drag the mouse. We had set 'x_view_step'
         * to 90 at the top of this file (where we declared all our variables).
         * 
         * We then add the horizontal change in camera angle to our 'x_view_angle'
         * variable, which keeps track of the cumulative horizontal change in our
         * camera angle. 'x_view_angle' is used in the camera rotations specified
         * in the 'display' function.
         */
        x_view_angle += ((float) x - (float) mouse_x) * mouse_scale_x * x_view_step;
        
        
        /* We do basically the same process as above to compute the vertical change
         * in camera angle. The only real difference is that we want to keep the
         * camera angle changes realistic, and it is unrealistic for someone in
         * real life to be able to change their vertical "camera angle" more than
         * ~90 degrees (they would have to detach their head and spin it vertically
         * or something...). So we decide to restrict the cumulative vertical angle
         * change between -90 and 90 degrees.
         */
        float temp_y_view_angle = y_view_angle +
                                  ((float) y - (float) mouse_y) * mouse_scale_y * y_view_step;
        y_view_angle = (temp_y_view_angle > 90 || temp_y_view_angle < -90) ?
                       y_view_angle : temp_y_view_angle;
        
        /* We update our 'mouse_x' and 'mouse_y' variables so that if the user moves
         * the mouse again without releasing it, then the distance we compute on the
         * next call to the 'mouse_moved' function will be from this current mouse
         * position.
         */
        mouse_x = x;
        mouse_y = y;
        
        /* Tell OpenGL that it needs to re-render our scene with the new camera
         * angles.
         */
        glutPostRedisplay();
    }
}


