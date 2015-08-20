#include <tracker.hpp>
#include <cv.h>

#include <opencv2\highgui\highgui.hpp>

class TrackerKiseleva : public Tracker
{
 public:
    virtual ~TrackerKiseleva() {}

    virtual bool init( const cv::Mat& frame, const cv::Rect& initial_position );
    virtual bool track( const cv::Mat& frame, cv::Rect& new_position );

 private:
    cv::Rect position_;
	cv::Mat previous_frame_;
};

bool TrackerKiseleva::init( const cv::Mat& frame, const cv::Rect& initial_position )
{
    position_ = initial_position;
	previous_frame_ = frame.clone();
	return true;
}

bool TrackerKiseleva::track( const cv::Mat& frame, cv::Rect& new_position )
{
	CV_Assert(previous_frame_.size() == frame.size());
    new_position = position_;
	
	// Get points to consider.
	std::vector<cv::Point2f> points_prev;
	const int m = 100;
	const double qLevel = 0.1;
	const double dist = 5.0;
	cv::Mat previous_frame_gray;
	cvtColor(previous_frame_(position_), previous_frame_gray, cv::COLOR_BGR2GRAY);
	cv::goodFeaturesToTrack(previous_frame_gray, points_prev, m, qLevel, dist);
	size_t n = points_prev.size();
	CV_Assert(n);
	for (size_t i = 0; i < n; ++i)
	{
		points_prev[i].x += position_.x;
		points_prev[i].y += position_.y;
	}
	
	// Compute optical flow in selected points.
	std::vector<uchar> state;
	std::vector<cv::Point2f> points;
	std::vector<float> error;
	cv::calcOpticalFlowPyrLK(previous_frame_, frame, points_prev, points, state, error);
	std::vector<cv::Point2f> tmp;
	std::vector<cv::Point2f> good_points;

	std::sort(error.begin(), error.end());
	double median_err = error[error.size()/2];

	for (size_t i = 0; i < n; ++i)
		if ((state[i])&&(error[i]<=median_err))
		{
			good_points.push_back(points_prev[i]);
			tmp.push_back(points[i]);

		}

	size_t s = good_points.size();
	CV_Assert(n == points.size());

	// Find points shift.
	std::vector<float> shifts_x(s);
	std::vector<float> shifts_y(s);

	for (size_t i = 0; i < s; ++i)
	{
		shifts_x[i] = tmp[i].x - good_points[i].x;
		shifts_y[i] = tmp[i].y - good_points[i].y;
	}
	std::sort(shifts_x.begin(), shifts_x.end());
	std::sort(shifts_y.begin(), shifts_y.end());
	// Find median shift.
	cv::Point2f median_shift(shifts_x[s / 2], shifts_y[s / 2]);

	new_position.x = static_cast<int>(position_.x + median_shift.x);
	new_position.y = static_cast<int>(position_.y + median_shift.y);
	previous_frame_ = frame.clone();
	return true;
}

cv::Ptr<Tracker> createTrackerKiseleva()
{
    return cv::Ptr<Tracker>(new TrackerKiseleva());
}
