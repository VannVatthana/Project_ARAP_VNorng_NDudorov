#include <igl/boundary_loop.h>
#include <igl/readOFF.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/unproject_onto_mesh.h>
#include <igl/point_mesh_squared_distance.h>
#include <igl/unproject.h>
#include <igl/project.h>
#include <igl/lscm.h>

#include "ARAPMethod.cpp"



Eigen::MatrixXd V;
Eigen::MatrixXi F;
ARAPMethod deformer;
std::vector<int> base_vertices,drag_vertices; //stable points and handle points
bool selectHandlePts = false;
bool selectStablePts = false;

double M_PI = 3.1415;

Eigen::MatrixXd C;

void Rotation(MatrixXd &V, RowVector3d u, double theta) 
{
    double w = cos(theta/2);
    RowVector3d im = (u / u.norm()) * sin(theta/2);
    Quaterniond q = Quaterniond(w, im[0],im[1],im[2]); 
    for(int i = 0; i < V.rows(); ++i)
    {
        RowVector3d point = V.row(i);
        Quaterniond rotated_point = q*Quaterniond(0, point[0],point[1],point[2])*q.inverse();
        V.row(i) = RowVector3d(rotated_point.x(), rotated_point.y(), rotated_point.z());
    }
}

int closestVertex(int face_id, Eigen::Vector3f& bc) // Find the vertex id from the face that is the closest to raycast
{
    Eigen::MatrixXd vCorr(3, 3); //recover the coordinates of the 3 vertices of the face
    for (int i = 0; i < 3; i++) {
        vCorr.row(i) = V.row(F(face_id, i));
    }

    // Get the result coordinate from raycast
    Eigen::Vector3d queryPt = (bc(0) * vCorr.row(0)) + (bc(1) * vCorr.row(1)) + (bc(2) * vCorr.row(2));
    Eigen::Vector3d dist;
    for (int i = 0; i < 3; i++) {
        Eigen::Vector3d diff = vCorr.row(i) - queryPt.transpose();
        dist(i) = std::sqrt(diff.dot(diff.transpose()));
    }

    int closest = 0;
    if (dist(0) <= dist(1) && dist(0) <= dist(2)) {
        closest = 0;
    }
    else if (dist(1) < dist(0) && dist(1) <= dist(2)) {
        closest = 1;
    }
    else if (dist(2) < dist(0) && dist(2) < dist(1)) {
        closest = 2;
    }
    return F(face_id, closest);
}

bool key_down(igl::opengl::glfw::Viewer& viewer, unsigned char key, int modifier) // Mode selection
{
    if (key == '1') // Handle points selection mode
    {
      selectHandlePts = !selectHandlePts;
      selectStablePts = false;

      if(selectHandlePts)
        std::cout << "Handle Points Selection Activated" <<std::endl;
      else
        std::cout << "Handle Points Selection Deactivated" <<std::endl;
    }
    else if (key == '2') // Stable points selection mode
    {
      selectHandlePts = false;
      selectStablePts = !selectStablePts;
      
      if(selectStablePts)
        std::cout << "Stable Points Selection Activated" <<std::endl;
      else
        std::cout << "Stable Points Selection Deactivated" <<std::endl;
    }

  // For rotation of handle points around their center
    else if (key == 'Q') // Rotate the handle points counter clockwise
    {
      MatrixXd positions(drag_vertices.size(), 3);
      for(int i = 0; i < drag_vertices.size(); i++)
      {
        positions.row(i) = V.row(drag_vertices.at(i));
      } 
      Vector3d rotation_axis(0,0,1);
      double angle = M_PI/16;
      Vector3d center = positions.colwise().mean();
      positions = positions.rowwise() - center.transpose();
      Rotation(positions,rotation_axis,angle);
      positions = positions.rowwise() + center.transpose();
      deformer.set_handle_points_positions(positions);
      V = deformer.get_V();
      viewer.data().set_mesh(V,F);
      viewer.data().compute_normals();
    }
    else if(key == 'E') // Rotate the handle points clockwise
    {
      MatrixXd positions(drag_vertices.size(), 3);
      for(int i = 0; i < drag_vertices.size(); i++)
      {
        positions.row(i) = V.row(drag_vertices.at(i));
      } 
      Vector3d rotation_axis(0,0,1);
      double angle = -M_PI/16;
      Vector3d center = positions.colwise().mean();
      positions = positions.rowwise() - center.transpose();
      Rotation(positions,rotation_axis,angle);
      positions = positions.rowwise() + center.transpose();
      deformer.set_handle_points_positions(positions);
      V = deformer.get_V();
      viewer.data().set_mesh(V,F);
      viewer.data().compute_normals();
    }
    // For deformation using keys
    else if (key == 'S')
    {
      MatrixXd positions(drag_vertices.size(), 3);
      for(int i = 0; i < drag_vertices.size(); i++)
      {
        positions.row(i) = V.row(drag_vertices.at(i));
      } 
      Vector3d shift(0,-1,0);
      positions = positions.rowwise() + shift.transpose();
      deformer.set_handle_points_positions(positions);
      V = deformer.get_V();
      viewer.data().set_mesh(V,F);
      viewer.data().compute_normals();
    }

    else if (key == 'W')
    {
      MatrixXd positions(drag_vertices.size(), 3);
        for(int i = 0; i < drag_vertices.size(); i++)
        {
          positions.row(i) = V.row(drag_vertices.at(i));
        } 
      Vector3d shift(0,1,0);
      positions = positions.rowwise() + shift.transpose();
      deformer.set_handle_points_positions(positions);
      V = deformer.get_V();
      viewer.data().set_mesh(V,F);
      viewer.data().compute_normals();
    }

    else if (key == 'A')
    {
      MatrixXd positions(drag_vertices.size(), 3);
      for(int i = 0; i < drag_vertices.size(); i++)
      {
        positions.row(i) = V.row(drag_vertices.at(i));
      } 
      Vector3d shift(-1,0,0);
      positions = positions.rowwise() + shift.transpose();
      deformer.set_handle_points_positions(positions);
      V = deformer.get_V();
      viewer.data().set_mesh(V,F);
      viewer.data().compute_normals();
    }

    else if (key == 'D')
    {
      MatrixXd positions(drag_vertices.size(), 3);
      for(int i = 0; i < drag_vertices.size(); i++)
      {
        positions.row(i) = V.row(drag_vertices.at(i));
      } 
      Vector3d shift(1,0,0);
      positions = positions.rowwise() + shift.transpose();
      deformer.set_handle_points_positions(positions);
      V = deformer.get_V();
      
      viewer.data().set_mesh(V,F);
      viewer.data().compute_normals();;
    }

  return false;
}
bool mouse_down(igl::opengl::glfw::Viewer& viewer, int mouseID, int modifier)
{
  double mx = viewer.current_mouse_x;
  double my = viewer.core().viewport(3) - viewer.current_mouse_y;
  int face_id;
  Eigen::Vector3f bc;

  if(selectHandlePts) 
  {
    if(igl::unproject_onto_mesh(Eigen::Vector2f(mx,my), viewer.core().view, viewer.core().proj, viewer.core().viewport, V, F, face_id, bc))
    {
      face_id = closestVertex(face_id, bc);
      std::vector<int>::iterator itr = std::find(drag_vertices.begin(), drag_vertices.end(), face_id);

      if(mouseID == 0) 
      {
        if(itr == drag_vertices.end())
        {
          C.row(face_id) = Vector3d(255,255,0); // update the color of the stable pts to yellow
          drag_vertices.push_back(face_id);
        }
      }
      else if(mouseID == 2)
      {
        
        if(itr != drag_vertices.end())
        { 
          C.row(face_id) = Vector3d(255,255,255); // back to white
          drag_vertices.erase(itr);
        }
      }
      else{
        return false;
      }
    }
    deformer.update_handle_points(drag_vertices); 
    viewer.data().set_colors(C);
    return true;
  }
  
  if(selectStablePts) 
  {
    if(igl::unproject_onto_mesh(Eigen::Vector2f(mx,my), viewer.core().view, viewer.core().proj, viewer.core().viewport, V, F, face_id, bc))
    {
      face_id = closestVertex(face_id, bc);
      std::vector<int>::iterator itr = std::find(base_vertices.begin(), base_vertices.end(), face_id);

      if(mouseID == 0) 
      {
        if(itr == base_vertices.end())
        {
          base_vertices.push_back(face_id);
          C.row(face_id) = Vector3d(255,0,0); // update the color of the stable pts to red
        }
      }
      else if(mouseID == 2)
      {
        
        if(itr != base_vertices.end())
        { 
          base_vertices.erase(itr);
          C.row(face_id) = Vector3d(255,255,255); // back to white
        }
      }
      else{
        return false;
      }
    }
    deformer.update_constraint_points(base_vertices); 
    viewer.data().set_colors(C);
    return true;
  }
  return false;
}

// This function is only when using bar2.off to predefine its constraints
void define_base(std::vector<int> &base_vertices, std::vector<int> &drag_vertices)
{
  int N = 5*5; //depends on mesh
  int n = 5*4 - 4;

  for(int v=0;v<N;++v)
  {
    base_vertices.push_back(v);
    drag_vertices.push_back(v+N);
  }
  for(int v=2*N;v<2*N+n;++v)
  {
    base_vertices.push_back(v);
  }
  for(int v = 0; v < n; ++v)
  {
    drag_vertices.push_back(V.rows()-1-v);
  }
}

void scale_mesh(MatrixXd &mesh)
{
  double scale_factor = 10/(V.colwise().maxCoeff() - V.colwise().minCoeff()).mean();
  Vector3d center = mesh.colwise().mean();
  mesh = mesh.rowwise() - center.transpose();
  mesh = mesh*scale_factor;
}

int main(int argc, char *argv[])
{
  using namespace Eigen;
  using namespace std;

  if(argc<2) {
		std::cout << "Error: input file required (.OFF)" << std::endl;
		return 0;
	}
	std::cout << "reading input file: " << argv[1] << std::endl;

  // Load a mesh in OFF format
  igl::readOFF(argv[1], V, F);
  scale_mesh(V);

  // Usage Instruction
  std::cout << "Press 1 : Activate/Deactivate the Handle Points Selection Mode " << endl
            << "Left Click to select or Right Click to unselect a vertex as a Handle Point" <<endl
            << "Press 2 : Activate/Deactivate the Stable Points Selection Mode" << endl
            << "Left Click to select or Right Click to unselect a vertex as a Stable Point" << endl
            << "Press W : Move up the handle points" <<endl
            << "Press S : Move down" <<endl
            << "Press A : Move to the left" << endl
            << "Press D : Move to the right" <<endl
            << "Press E : Rotate handle points clockwise around the center of all the handle points" <<endl
            << "Press Q : Rotate handle points counter-clockwise" <<endl; 
  // Plot the mesh
  igl::opengl::glfw::Viewer viewer;
  viewer.data().set_mesh(V, F);
  viewer.callback_key_down = &key_down;
  viewer.callback_mouse_down = &mouse_down;

  // Disable wireframe
  viewer.data().show_lines = true;

  // Draw checkerboard texture
  viewer.data().show_texture = true;

  // Only for bar2.off  
  //define_base(base_vertices,drag_vertices);

  // create deformer
  SparseLU<SparseMatrix<double>, COLAMDOrdering<int>> solver;
  deformer = ARAPMethod(V, F, base_vertices, drag_vertices, &solver);

  // visualisation of mouvable and dragged vertices
  C = MatrixXd (V.rows(),3);
  C.setConstant(255);
  for(int i=0; i<base_vertices.size();++i)
      C.row(base_vertices[i])=Vector3d(255,0,0);
  for(int i=0; i<drag_vertices.size();++i)
      C.row(drag_vertices[i])=Vector3d(255,255,0);
  // Assign per-vertex colors
	viewer.data().set_colors(C);

  // Launch the viewer
  viewer.launch();
}