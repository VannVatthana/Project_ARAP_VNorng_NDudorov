#include <igl/opengl/glfw/Viewer.h>
#include <igl/cotmatrix.h>
#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>
using namespace Eigen;
using namespace std;



class ARAPMethod
{
public :
	ARAPMethod(){}
	ARAPMethod(const MatrixXd &vertices, const MatrixXi &faces, 
			   const vector<int> &constraint_points, const vector<int> &handle_points,
			   SparseLU<SparseMatrix<double>, COLAMDOrdering<int>>* system_solver) 
	{
		F = faces;
		V = vertices;
		V_prime = vertices;
		constraintPts = constraint_points;
		handlePts = handle_points;
		solver = system_solver;

		nVertices = V.rows();
		nFaces = F.rows();
		neighborsOfVertices = findNeighbors(nVertices, F);
		set_W(V); //define weight W matrix on input mesh
		set_L(W, constraintPts, handlePts); //define linear system matrix L
		set_solver(L); //precompute solver for system L
	}

	void update_handle_points(const vector<int> &handle_points)
	{
		handlePts=handle_points;
		set_L(W, constraintPts, handlePts);
		set_solver(L);
	}

	void update_constraint_points(const vector<int> &constraint_points)
	{
		constraintPts=constraint_points;
		set_L(W, constraintPts, handlePts);
		set_solver(L);
	}

	void set_handle_points_positions(const MatrixXd &handle_points_positions)
	{	
		deformation(handle_points_positions);
	}

	MatrixXd get_V()
	{
		return V;
	}

private :
	MatrixXd V; // Vertices of mesh
	MatrixXd V_prime; // Vertices of deformed mesh
	MatrixXi F; // Faces of mesh
	vector<vector<int>> neighborsOfVertices; // each subvector contains the neighbors of each vertex

	vector<int> handlePts; // Vertices whose positions are defined by user
	vector<int> constraintPts; // Vertices that do not deform

	SparseMatrix<double> W; // Matrix of weight of edge between two vertices (i, j)
	SparseMatrix<double> L; //right side of Lp' = b
	MatrixXd b; //left side of Lp' = b
	vector<Matrix3d> cellRotationMatrices;
	SparseLU<SparseMatrix<double>, COLAMDOrdering<int>>* solver;

	int nVertices;
	int nFaces;

	vector<vector<int>> findNeighbors(int nV, const MatrixXi &F) 
	{	
		vector<vector<int>> neighborsVect(nV);
		for (int p = 0; p < F.rows(); p++)
		{
			int i = F(p, 0);
			int j = F(p, 1);
			int k = F(p, 2);

			if (!count(neighborsVect.at(i).begin(), neighborsVect.at(i).end(), j))
				neighborsVect.at(i).push_back(j);

			if (!count(neighborsVect.at(i).begin(), neighborsVect.at(i).end(), k))
				neighborsVect.at(i).push_back(k);

			if (!count(neighborsVect.at(j).begin(), neighborsVect.at(j).end(), k))
				neighborsVect.at(j).push_back(k);
			
			if (!count(neighborsVect.at(j).begin(), neighborsVect.at(j).end(), i))
				neighborsVect.at(j).push_back(i);

			if (!count(neighborsVect.at(k).begin(), neighborsVect.at(k).end(), i))
				neighborsVect.at(k).push_back(i);

			if (!count(neighborsVect.at(k).begin(), neighborsVect.at(k).end(), j))
				neighborsVect.at(k).push_back(j);
		}
		return neighborsVect;
	}
	
	// Set constraint vertices, including static and handle vertices
	void setConstraintVertices(const MatrixXd &V, const vector<int> &constraint_points)
	{
		for(int c : constraint_points)
			V_prime.row(c) = V.row(c); // Suppose for now that all the constraint vertices are stable vertices
	}
	
	void setHandleConstraint(const MatrixXd &handle_positions, const vector<int> &handle_points)
	{
		if(handle_positions.rows() != handle_points.size())
		{
			throw std::invalid_argument("handle points number and number of handle points positions do not match");
		}
		for(int i = 0; i < handle_points.size(); ++i)
		{
			V_prime.row(handle_points[i]) = handle_positions.row(i);
		}
	}

	void set_W(const MatrixXd &V) //define per-edge-weight matrix
	{
		W = SparseMatrix<double>(nVertices,nVertices);
		igl::cotmatrix(V,F,W); // cotan weights
		W = 0.5*W;
	}

	void set_cellRotations(const MatrixXd &V, const MatrixXd &V_prime, const SparseMatrix<double> &W)
	{
		vector<Matrix3d> allRotationMatrices(nVertices);
		for (int i = 0; i < nVertices; i++) 
		{
			Matrix3d rotate = Matrix3d::Zero();
			vector<int> i_neighbors = neighborsOfVertices.at(i);
			// calculate covariance matrix S = PDP'
			int nn = i_neighbors.size();
			MatrixXd P = MatrixXd::Zero(3, nn);
			MatrixXd P_prime = MatrixXd::Zero(3, nn);
			MatrixXd D = MatrixXd::Zero(nn, nn);

			for (int k = 0; k < nn; k++) {
				int j = i_neighbors.at(k);
				P.col(k) = (V.row(i) - V.row(j)).transpose();
				P_prime.col(k) = (V_prime.row(i) - V_prime.row(j)).transpose(); 
				D(k, k) = W.coeff(i, j);
			}
			MatrixXd covMatrix = P * D * P_prime.transpose();

			JacobiSVD<MatrixXd> svd(covMatrix, ComputeFullU | ComputeFullV);

			MatrixXd v = svd.matrixV();
			MatrixXd u = svd.matrixU();
			if ((v*u.transpose()).determinant()<0)
			{	
				u.rightCols(1) = u.rightCols(1) * (-1);
			}

			rotate = v * u.transpose();
			
			allRotationMatrices.at(i) = rotate;
		}
		cellRotationMatrices=allRotationMatrices;
	}
	
	void set_L(const SparseMatrix<double> &W, 
	           const vector<int> &constraintPts, const vector<int> &handlePts) // set left hand side of the equation system Lp'=b
	{
		L = SparseMatrix<double>(nVertices,nVertices);
		vector<Triplet<double>> L_triplets;
		for (int i = 0; i < nVertices; i++)
		{
			if((count(constraintPts.begin(),constraintPts.end(),i)>0)
			 ||(count(handlePts.begin(),handlePts.end(),i)>0))
			{
				L_triplets.push_back(Triplet<double>(i, i, 1));
			}
			else 
			{
				double L_ii = 0;
				for (int j : neighborsOfVertices.at(i))
				{
					L_ii += W.coeff(i, j);
					L_triplets.push_back(Triplet<double>(i, j, -W.coeff(i, j)));
				}
				L_triplets.push_back(Triplet<double>(i, i, L_ii));
			}			
		}
		L.setFromTriplets(L_triplets.begin(),L_triplets.end());
	}

	// set the right hand side of the equation system Lp'=b
	void set_b(const MatrixXd &V, const MatrixXd &V_prime, const SparseMatrix<double> &W, const vector<Matrix3d> &cellRotationMatrices,
	           const vector<int> &constraintPts, const vector<int> &handlePts) 
	{
		b = MatrixXd::Zero(nVertices, 3);
		for (int i = 0; i < nVertices; i++)
		{
			if((count(constraintPts.begin(),constraintPts.end(),i)>0)
			 ||(count(handlePts.begin(),handlePts.end(),i)>0))
			{
				b.row(i) = V_prime.row(i);
			}
			else
			{
				for (int j : neighborsOfVertices.at(i))
				{
					b.row(i) += (W.coeff(i, j) / 2.) * (cellRotationMatrices.at(i) + cellRotationMatrices.at(j)) * (V.row(i) - V.row(j)).transpose();
				}
			}
		}	
	}

	void set_solver(const SparseMatrix<double> &L)
	{
		// Compute the ordering permutation vector from the structural pattern of L
		(*solver).analyzePattern(L); 
		// Compute the numerical factorization 
		(*solver).factorize(L); 
	}
	
	// Estimate new positions for the deformed vertices
	// New position of vertices V_prime is obtained by solving Lp' = b
	MatrixXd estimateNewPositionOfVertices(const SparseLU<SparseMatrix<double>, COLAMDOrdering<int>> &solver, 
	                                       const MatrixXd &b) 
	{
		MatrixXd NewPositions(b.rows(),3);

		//Use the factors to solve the linear system 
		NewPositions.col(0) = solver.solve(b.col(0));  // x cord
		NewPositions.col(1) = solver.solve(b.col(1));  // y cord
		NewPositions.col(2) = solver.solve(b.col(2));  // z cord

		return NewPositions;
	}

	void deformation(const MatrixXd &handle_points_positions)
	{
		setConstraintVertices(V, constraintPts);
		setHandleConstraint(handle_points_positions, handlePts);

		int iter = 0;
		int maxIter = 100;
		MatrixXd V_previous=V_prime;
		double delta = 1e-3;

		// iterate untill ||v_i - v_i+1|| <= 'delta'
		while(true)
		{
			//compute Rotations from first guess positions
			set_cellRotations(V, V_prime, W); 

			//compute new positions using calculated Rotations 
			set_b(V, V_prime, W, cellRotationMatrices, constraintPts, handlePts);
			V_prime = estimateNewPositionOfVertices(*solver, b);

			// check if convergence is obtained
			if((V_prime-V_previous).norm() <= delta)
				break;

			// if(iter % 10 == 0)
			// 	std::cout<< (V_prime-V_previous).norm() << " " << iter << std::endl;
			
			V_previous = V_prime;
			++iter;
			if(iter >= maxIter)
				break;
		}
		V = V_prime; // V and V_prime are again the same and need to get new handle points positions to deform mesh
	}
};