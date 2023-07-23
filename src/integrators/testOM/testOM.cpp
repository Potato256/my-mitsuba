#include <mitsuba/render/scene.h>
#include "myOM.h"

MTS_NAMESPACE_BEGIN

#define OMSIZE 32
#define OMDEPTH OMSIZE / 32

class TestOMIntegrater : public SamplingIntegrator
{
public:
    MTS_DECLARE_CLASS()
public:
    /// Initialize the integrator with the specified properties
    TestOMIntegrater(const Properties &props) : SamplingIntegrator(props)
    {
        Spectrum defaultColor;
        defaultColor.fromLinearRGB(1.f, 1.f, 1.f);
        m_color = props.getSpectrum("color", defaultColor);
    }

private:
    Spectrum m_color;
    Float m_maxDist;
    AABB m_baseAABB;
    OccupancyMap<OMSIZE, OMDEPTH> m_om;
    OccupancyMap<OMSIZE, OMDEPTH> test_om;

public:
    /// Unserialize from a binary data stream
    TestOMIntegrater(Stream *stream, InstanceManager *manager)
        : SamplingIntegrator(stream, manager) {}

    /// Serialize to a binary data stream
    void serialize(Stream *stream, InstanceManager *manager) const
    {
        SamplingIntegrator::serialize(stream, manager);
    }

    /// Preprocess function -- called on the initiating machine
    bool preprocess(const Scene *scene, RenderQueue *queue,
                    const RenderJob *job, int sceneResID, int cameraResID,
                    int samplerResID)
    {
        SamplingIntegrator::preprocess(scene, queue, job, sceneResID,
                                       cameraResID, samplerResID);

        const AABB &sceneAABB = scene->getAABB();
        /* Find the camera position at t=0 seconds */
        Point cameraPosition = scene->getSensor()->getWorldTransform()->eval(0).transformAffine(Point(0.0f));
        m_maxDist = -std::numeric_limits<Float>::infinity();

        for (int i = 0; i < 8; ++i)
            m_maxDist = std::max(m_maxDist,
                                 (cameraPosition - sceneAABB.getCorner(i)).length());

        Point m_min(1e30f), m_max(-1e30f), m_center, m_lcorner;
        auto meshes = scene->getMeshes();

        for (auto m : meshes){
            // SLog(EInfo, m->toString().c_str());
            m_min.x = std::min(m_min.x, m->getAABB().min.x);
            m_min.y = std::min(m_min.y, m->getAABB().min.y);
            m_min.z = std::min(m_min.z, m->getAABB().min.z);
            m_max.x = std::max(m_max.x, m->getAABB().max.x);
            m_max.y = std::max(m_max.y, m->getAABB().max.y);
            m_max.z = std::max(m_max.z, m->getAABB().max.z);
        }

        Vector3 d = m_max - m_min;
        Float r = 0;
        for (int i = 0; i < 3; ++i)
            r = std::max(r, d[i]);
        r *= 0.5 * 1.01f;
        m_center = m_min + d / 2.0f;
        m_lcorner = m_center - Vector3(r);
        m_baseAABB = AABB(m_lcorner, m_lcorner + Vector3(2 * r));

        m_om.clear();
        m_om.setAABB(m_baseAABB);
        m_om.setSize(2 * r);
        // m_om.testSetAll();
        // m_om.testSetBallPattern();
        m_om.setScene(scene);
        test_om.clear();
        test_om.setAABB(m_baseAABB);
        test_om.setSize(2 * r);
        m_om.generateROMA(&test_om, Point2(0.2, 0.2));

        // std::ostringstream oss;
        // oss<<
        // SLog(EDebug, m_om.toString().c_str());
        // SLog(EDebug, m_om.toString().c_str());

        return true;
    }

    /// Query for an unbiased estimate of the radiance along <tt>r</tt>
    Spectrum Li(const RayDifferential &r, RadianceQueryRecord &rRec) const
    {

        Float nearT;
        if (test_om.rayIntersect(r, nearT))
        {
            return Spectrum(0.8f - nearT / m_maxDist) * m_color;
        }
        // if (rRec.rayIntersect(r)) {
        //     Float distance = rRec.its.t;
        //     return Spectrum(1.0f - distance/m_maxDist) * m_color;
        // }
        return Spectrum(0.0f);
    }
};

MTS_IMPLEMENT_CLASS_S(TestOMIntegrater, false, SamplingIntegrator)

MTS_EXPORT_PLUGIN(TestOMIntegrater, "Occupancy map test");

MTS_NAMESPACE_END