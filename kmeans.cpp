#include "kmeans.hpp"
#include <queue>
#include <cassert>
#include <limits>
#include <thread>

std::istream &operator>>(std::istream &is, Point &pt) {
  return is >> pt.x >> pt.y;
}

std::ostream &operator<<(std::ostream &os, Point &pt) {
  return os << pt.x << " " << pt.y;
}

Kmeans::Kmeans(const std::vector<Point> &points,
               const std::vector<Point> &init_centers) {
  m_points = points;
  m_centers = init_centers;
  m_numPoints = points.size();
  m_numCenters = init_centers.size();
}

std::vector<index_t> Kmeans::Run(int max_iterations) {
  std::vector<index_t> assignment(m_numPoints, 0); // the return vector
  int curr_iteration = 0;
  std::cout << "Running kmeans with num points = " << m_numPoints
            << ", num centers = " << m_numCenters
            << ", max iterations = " << max_iterations << "...\n";

  std::vector<int> p_cnt(m_numCenters, 0);

  while (max_iterations--) {
    ++curr_iteration;
    std::vector<index_t> assignment_bak = assignment;

    #pragma omp parallel for 
    for (int i = 0; i < m_numPoints; ++i) {
      Point &p_i = m_points[i];
      double min_dis = std::numeric_limits<double>::max();
      int place = -1;
      for (int k = 0; k < m_numCenters; ++k) {
        double dis = (p_i.x - m_centers[k].x) * (p_i.x - m_centers[k].x) + (p_i.y - m_centers[k].y) * (p_i.y - m_centers[k].y);
        if (dis < min_dis) {
          min_dis = dis;
          place = k;
        }
      }
      assignment[i] = place;
    }

    if (assignment == assignment_bak) {
      goto converge;
    }

    Point p;
    m_centers.assign(m_numCenters, p);
    p_cnt.assign(m_numPoints, 0);

    for (int i = 0; i < m_numPoints; ++i) {
      index_t cluster_1 = assignment[i];
      m_centers[cluster_1].x += m_points[i].x;
      m_centers[cluster_1].y += m_points[i].y;
      p_cnt[cluster_1]++;
    }

    for (int j = 0; j < m_numCenters; ++j) {
      m_centers[j].x /= p_cnt[j];
      m_centers[j].y /= p_cnt[j];
    }
    
  }

converge:
  std::cout << "Finished in " << curr_iteration << " iterations." << std::endl;
  return assignment;
}
